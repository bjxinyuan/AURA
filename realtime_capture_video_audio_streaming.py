"""
Realtime Video Audio Capture Client - Streaming Version

支持 Qwen3_VL_online_streaming.py 的流式输入/输出协议。

Downstream messages from the main inference service are fanned out
from a background TCP receive thread onto a multiplexed event_queue,
and the browser subscribes via /api/events (SSE).

协议定义:
- Type 1: VIDEO (C->S)
- Type 2: AUDIO (C->S)
- Type 4: CLEAR_CONTEXT (C->S)
- Type 6: START_CAMERA (C->S)
- Type 7: ERROR (S->C) - 服务端错误
- Type 8: STREAMING_TOKEN (S->C) - 流式 token
- Type 9: TTS_AUDIO_CHUNK (S->C) - TTS 流式 PCM chunk
- Type 10: ASR_QUERY_ECHO (S->C) - ASR 转写文字回显
"""

import argparse
import os
import base64
import struct
import socket
import threading
import time
import queue

from aura.sse import format_sse_stream
from collections import deque
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 服务端配置 (实际值由 main() 中 argparse + AURA_INFER_HOST/AURA_INFER_PORT 注入)
SERVER_HOST = None
SERVER_PORT = None

# 视频发送间隔（秒） - 改为连续发送
VIDEO_SEND_INTERVAL = 1.0
# VIDEO_BUFFER_DURATION = 2.0  # 缓存最近 2 秒的视频（共约 4 帧）
# video_buffer = deque()  # 存储 (timestamp, data)
last_video_send_time = 0

# 协议类型
VIDEO_TYPE = b'\x01'
AUDIO_TYPE = b'\x02'
CLEAR_CONTEXT_TYPE = b'\x04'  # 清空上下文
START_CAMERA_TYPE = b'\x06'  # 开启摄像头（清理文件夹）
ERROR_TYPE = 7  # 服务器错误/拒绝消息
STREAMING_TOKEN_TYPE = 8  # 流式 token
TTS_AUDIO_CHUNK_TYPE = 9  # TTS 音频 chunk (Raw PCM int16)
ASR_QUERY_ECHO_TYPE = 10  # ASR query echo (Plan 2: 立即回传用户转写文字)

# 全局socket连接
socket_lock = threading.Lock()
global_socket = None
last_connect_attempt = 0
connect_retry_interval = 5
connection_error_logged = False

# Multiplexed event queue drained by /api/events (SSE). Each item is a
# (kind, payload_dict) tuple. Kinds: "token", "chunk", "error", "close".
event_queue = queue.Queue()

# 用户会话锁：确保同时只有一个浏览器用户可以使用系统
user_session_lock = threading.Lock()
current_user_session = None  # 当前占用会话的用户 session_id

def verify_session(session_id: str) -> bool:
    """验证 session_id 是否是当前活动会话。"""
    with user_session_lock:
        return current_user_session is not None and current_user_session == session_id

def get_session_error_response():
    """返回会话验证失败的标准错误响应。"""
    return jsonify({
        'success': False,
        'error': '会话无效或已过期，请重新开启摄像头'
    }), 401

def get_socket():
    """获取或创建socket连接"""
    global global_socket, last_connect_attempt, connection_error_logged
    
    current_time = time.time()
    
    with socket_lock:
        if global_socket is not None:
            return global_socket
        
        # 检查重试间隔
        if current_time - last_connect_attempt < connect_retry_interval:
            return None
        
        last_connect_attempt = current_time
        
        try:
            global_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            global_socket.settimeout(None)  # 设为阻塞模式，由接收线程处理
            global_socket.connect((SERVER_HOST, SERVER_PORT))
            logger.info(f"✓ 已连接到服务端 {SERVER_HOST}:{SERVER_PORT}")
            connection_error_logged = False
            
            # 连接成功后，启动接收线程
            start_receive_thread()
            
            return global_socket
        except Exception as e:
            if not connection_error_logged:
                logger.warning(f"⚠ 服务端 {SERVER_HOST}:{SERVER_PORT} 不可用: {e}")
                logger.info(f"   将每 {connect_retry_interval} 秒重试连接...")
                connection_error_logged = True
            global_socket = None
            return None

def receive_thread_func():
    """后台接收线程"""
    global global_socket
    logger.info("启动后台接收线程")
    
    while True:
        sock = None
        with socket_lock:
            sock = global_socket
        
        if sock is None:
            time.sleep(1)
            continue
            
        try:
            # 读取头部
            header = sock.recv(9)
            if not header:
                logger.warning("服务端关闭了连接")
                with socket_lock:
                    if global_socket == sock:
                        global_socket.close()
                        global_socket = None
                continue
                
            msg_type, msg_len = struct.unpack('>BQ', header)
            
            # 读取内容
            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(msg_len - len(data), 4096))
                if not chunk:
                    break
                data += chunk
                
            if msg_type == ERROR_TYPE:
                # 服务器错误/拒绝消息
                error_msg = data.decode('utf-8')
                logger.warning(f"⚠️ 服务端错误: {error_msg}")
                event_queue.put(("error", {"message": error_msg}))
                # 关闭连接，因为服务器已拒绝
                with socket_lock:
                    if global_socket == sock:
                        try:
                            global_socket.close()
                        except:
                            pass
                        global_socket = None
                break  # 退出接收循环
            
            elif msg_type == STREAMING_TOKEN_TYPE:
                # Streaming model output → SSE "token" event; payload.raw
                # is the original JSON-encoded token envelope from backend.
                try:
                    token_data = data.decode('utf-8')
                    logger.debug(f"📝 收到流式 token: {token_data[:50]}...")
                    event_queue.put(("token", {"raw": token_data}))
                except Exception as e:
                    logger.error(f"解析流式 token 失败: {e}")
            
            elif msg_type == ASR_QUERY_ECHO_TYPE:
                # Plan 2: ASR query echo — share the "token" channel because
                # the frontend routes both through handleStreamingToken.
                try:
                    token_data = data.decode('utf-8')
                    logger.info(f"📤 收到 ASR query echo: {token_data[:50]}...")
                    event_queue.put(("token", {"raw": token_data}))
                except Exception as e:
                    logger.error(f"解析 ASR query echo 失败: {e}")

            elif msg_type == TTS_AUDIO_CHUNK_TYPE:
                # 收到 TTS 音频 chunk (Raw PCM int16) - Step 2 新增
                # 协议: response_id_len(1) + response_id + sentence_idx(2) + chunk_idx(2) + sample_rate(4) + is_final(1) + pcm_data
                try:
                    response_id_len = data[0]
                    response_id = data[1:1+response_id_len].decode('utf-8')
                    offset = 1 + response_id_len
                    sentence_idx, chunk_idx, sample_rate, is_final = struct.unpack(">HHIB", data[offset:offset+9])
                    pcm_data = data[offset+9:]
                    
                    if is_final:
                        logger.info(f"🔊 收到 TTS chunk [final] sentence={sentence_idx}")
                    else:
                        # 每10个chunk打印一次，避免日志过多
                        if chunk_idx % 10 == 0:
                            logger.info(f"🔊 收到 TTS chunk sentence={sentence_idx} chunk={chunk_idx} ({len(pcm_data)} bytes)")
                    
                    event_queue.put(("chunk", {
                        "response_id": response_id,
                        "sentence_idx": sentence_idx,
                        "chunk_idx": chunk_idx,
                        "sample_rate": sample_rate,
                        "is_final": bool(is_final),
                        "pcm_base64": base64.b64encode(pcm_data).decode("ascii") if pcm_data else "",
                    }))
                except Exception as parse_e:
                    logger.error(f"解析 TTS 音频 chunk 协议失败: {parse_e}")
                
        except Exception as e:
            logger.error(f"接收线程错误: {e}")
            with socket_lock:
                if global_socket == sock:
                    try:
                        global_socket.close()
                    except:
                        pass
                    global_socket = None
            time.sleep(1)

_receive_thread_started = False
def start_receive_thread():
    global _receive_thread_started
    if not _receive_thread_started:
        t = threading.Thread(target=receive_thread_func, daemon=True)
        t.start()
        _receive_thread_started = True

def send_data(data_type: bytes, data: bytes):
    """发送数据到服务端 (非阻塞，不等待响应)"""
    global global_socket
    
    try:
        sock = get_socket()
        if sock is None:
            return False
        
        # 构造消息: 类型(1字节) + 长度(8字节) + 数据
        message = data_type + struct.pack('>Q', len(data)) + data
        
        with socket_lock:
            sock.sendall(message)
            logger.info(f"✓ 已发送 {len(data)} 字节 ({'视频' if data_type==VIDEO_TYPE else '音频'})")
            return True
            
    except Exception as e:
        logger.error(f"发送数据失败: {e}")
        with socket_lock:
            if global_socket:
                try:
                    global_socket.close()
                except:
                    pass
                global_socket = None
        return False

@app.route('/')
def index():
    """主页面 - 使用 streaming 版本的模板"""
    return render_template('index_streaming.html', interval=VIDEO_SEND_INTERVAL)

@app.route('/api/video', methods=['POST'])
def receive_video():
    """接收视频帧并根据时间间隔判断是否发送"""
    global last_video_send_time
    try:
        session_id = request.form.get('session_id')
        if not verify_session(session_id):
            return get_session_error_response()
        
        if 'frame' not in request.files:
            return jsonify({'success': False, 'error': '没有视频数据'})
        
        frame_file = request.files['frame']
        frame_data = frame_file.read()
        
        # 立即检查是否需要发送
        current_time = time.time()
        sent = False
        
        # 简单的限流逻辑：如果距离上次发送超过间隔，则发送
        if current_time - last_video_send_time >= VIDEO_SEND_INTERVAL:
            success = send_data(VIDEO_TYPE, frame_data)
            if success:
                last_video_send_time = current_time
                sent = True
                logger.debug(f"📹 自动发送视频帧 (间隔 {VIDEO_SEND_INTERVAL}s)")
        
        return jsonify({
            'success': True,
            'size': len(frame_data),
            'sent': sent
        })
        
    except Exception as e:
        logger.error(f"处理视频帧失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/audio', methods=['POST'])
def receive_audio():
    """接收音频数据并发送给服务端"""
    try:
        session_id = request.form.get('session_id')
        if not verify_session(session_id):
            return get_session_error_response()
            
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '没有音频数据'})
            
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        success = send_data(AUDIO_TYPE, audio_data)
        
        if success:
            logger.info(f"🎤 发送音频数据 ({len(audio_data)} bytes)")
        
        return jsonify({
            'success': success,
            'size': len(audio_data)
        })
        
    except Exception as e:
        logger.error(f"处理音频失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取连接状态"""
    connected = global_socket is not None
    return jsonify({
        'connected': connected,
        'server': f"{SERVER_HOST}:{SERVER_PORT}"
    })

@app.route('/api/events')
def events():
    """Multiplexed Server-Sent Events stream — replaces poll_streaming_token,
    poll_tts_audio_chunk, and poll_error in one persistent connection."""
    session_id = request.args.get('session_id', '')
    if not verify_session(session_id):
        def _error_stream():
            yield ": session-invalid\n\n"
            yield 'event: error\ndata: {"message": "invalid session"}\n\n'
        return Response(_error_stream(), mimetype='text/event-stream')

    return Response(
        format_sse_stream(event_queue, heartbeat_seconds=15.0),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/api/acquire_session', methods=['POST'])
def acquire_session():
    """获取用户会话锁"""
    global current_user_session
    import uuid
    
    with user_session_lock:
        if current_user_session is None:
            new_session_id = str(uuid.uuid4())[:8]
            current_user_session = new_session_id
            
            # 清空残留的事件队列
            global event_queue
            event_queue = queue.Queue()

            logger.info(f"✅ 用户获取会话锁: {new_session_id}，已清空残留队列")
            return jsonify({
                'success': True,
                'session_id': new_session_id
            })
        else:
            logger.warning(f"⚠️ 会话被拒绝: 系统正被 {current_user_session} 使用")
            return jsonify({
                'success': False,
                'error': '系统正在被其他用户使用中，请点击"释放会话"按钮后重试'
            }), 423

@app.route('/api/force_release_session', methods=['POST'])
def force_release_session():
    """强制释放会话锁"""
    global current_user_session
    
    with user_session_lock:
        if current_user_session is not None:
            old_session = current_user_session
            current_user_session = None
            # Tell any open SSE stream to wind down cleanly.
            event_queue.put(("close", {}))
            logger.info(f"🔓 会话已释放: {old_session}")
            return jsonify({
                'success': True,
                'message': f'会话已释放'
            })
        else:
            return jsonify({
                'success': True,
                'message': '没有活动会话'
            })

@app.route('/api/session_status', methods=['GET'])
def session_status():
    """获取当前会话状态"""
    with user_session_lock:
        return jsonify({
            'occupied': current_user_session is not None
        })

@app.route('/api/clear_media', methods=['POST'])
def clear_media():
    """清理服务器端的视频和音频文件"""
    if request.is_json:
        session_id = request.json.get('session_id')
    else:
        session_id = request.form.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    import os
    import glob
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    VIDEO_DIR = os.path.join(base_dir, "real_time_captured_video")
    AUDIO_DIR = os.path.join(base_dir, "real_time_captured_audio")
    
    deleted_videos = 0
    deleted_audio = False
    
    try:
        # 清理视频文件
        if os.path.exists(VIDEO_DIR):
            video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4")) + \
                         glob.glob(os.path.join(VIDEO_DIR, "*.webm")) + \
                         glob.glob(os.path.join(VIDEO_DIR, "*.tmp"))
            for f in video_files:
                try:
                    os.remove(f)
                    deleted_videos += 1
                except Exception as e:
                    logger.warning(f"删除视频文件失败 {f}: {e}")
            
            merged_dir = os.path.join(VIDEO_DIR, "merged")
            if os.path.exists(merged_dir):
                merged_files = glob.glob(os.path.join(merged_dir, "*.mp4"))
                for f in merged_files:
                    try:
                        os.remove(f)
                        deleted_videos += 1
                    except Exception as e:
                        logger.warning(f"删除合并视频失败 {f}: {e}")
        
        # 清理音频文件
        audio_file = os.path.join(AUDIO_DIR, "latest.mp3")
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                deleted_audio = True
            except Exception as e:
                logger.warning(f"删除音频文件失败: {e}")
        
        logger.info(f"🗑 已清理媒体文件: {deleted_videos} 个视频, {'1' if deleted_audio else '0'} 个音频")
        
        # 发送清空上下文命令到服务端
        context_cleared = send_data(CLEAR_CONTEXT_TYPE, b'clear')
        
        return jsonify({
            'success': True,
            'deleted_videos': deleted_videos,
            'deleted_audio': deleted_audio,
            'context_cleared': context_cleared
        })
        
    except Exception as e:
        logger.error(f"清理媒体文件失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """开启摄像头时调用，通知服务端清理所有文件夹"""
    if request.is_json:
        session_id = request.json.get('session_id')
    else:
        session_id = request.form.get('session_id')
    if not verify_session(session_id):
        return get_session_error_response()
    
    try:
        logger.info("📷 开启摄像头，通知服务端清理文件夹")
        
        success = send_data(START_CAMERA_TYPE, b'start')
        if success:
            logger.info("✓ 已发送开启摄像头命令")
        else:
            logger.warning("⚠ 开启摄像头命令发送失败")
        
        return jsonify({
            'success': True,
            'command_sent': success
        })
        
    except Exception as e:
        logger.error(f"发送开启摄像头命令失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def main():
    """主函数"""
    global SERVER_HOST, SERVER_PORT

    parser = argparse.ArgumentParser(
        description="AURA Flask bridge — accepts video/audio from browser and forwards to inference server")
    parser.add_argument('--infer-host',
                        default=os.environ.get('AURA_INFER_HOST', '127.0.0.1'),
                        help='Inference server host (env: AURA_INFER_HOST, default: 127.0.0.1)')
    parser.add_argument('--infer-port', type=int,
                        default=int(os.environ.get('AURA_INFER_PORT', '12345')),
                        help='Inference server TCP port (env: AURA_INFER_PORT, default: 12345)')
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('AURA_FLASK_PORT', '5003')),
                        help='Flask listen port (env: AURA_FLASK_PORT, default: 5003)')
    parser.add_argument('--https', '-s', action='store_true',
                        help='Serve over HTTPS using cert.pem/key.pem next to this script')
    parser.add_argument('--tunnel', '-t', action='store_true',
                        help='Expose via Cloudflare Tunnel (requires pycloudflared)')
    args = parser.parse_args()

    SERVER_HOST = args.infer_host
    SERVER_PORT = args.infer_port
    use_https = args.https
    use_tunnel = args.tunnel
    port = args.port
    
    cert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cert.pem')
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'key.pem')
    has_certs = os.path.exists(cert_file) and os.path.exists(key_file)
    
    print("=" * 50)
    print("🎥 实时视频音频捕获客户端 (Streaming 版本)")
    print("=" * 50)
    print(f"📡 服务端地址: {SERVER_HOST}:{SERVER_PORT}")
    print("🔄 支持流式 Token 输出")
    
    tunnel_url = None
    if use_tunnel:
        try:
            from pycloudflared import try_cloudflare
            print("🌐 正在启动 Cloudflare Tunnel...")
            tunnel_url = try_cloudflare(port=port, verbose=False).tunnel
            print(f"✅ Cloudflare Tunnel 已启动!")
            print(f"🔗 公网访问地址: {tunnel_url}")
        except ImportError:
            print("❌ pycloudflared 未安装，请运行: pip install pycloudflared")
            use_tunnel = False
        except Exception as e:
            print(f"❌ Cloudflare Tunnel 启动失败: {e}")
            use_tunnel = False
    
    if use_https and has_certs:
        print("🔒 HTTPS 模式")
        print(f"🌐 访问地址: https://192.168.x.x:{port}")
    elif not use_tunnel:
        print("🌐 HTTP 模式")
        print(f"🌐 本机访问: http://localhost:{port}")
    
    print("=" * 50)
    print("功能说明:")
    print("  1. 视频: 每2秒发送一次")
    print("  2. 音频: 按住麦克风按钮录制，松开后发送")
    print("  3. 流式输出: 实时显示生成的 token")
    print("=" * 50)
    
    if use_https and has_certs:
        ssl_context = (cert_file, key_file)
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, ssl_context=ssl_context)
    else:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == '__main__':
    main()
