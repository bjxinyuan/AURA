// aura frontend — camera, video frame capture, and outbound video stream.

async function toggleVideo() {
    if (isVideoStreaming) {
        await stopVideo();
    } else {
        await startVideo();
    }
}

function updateVideoMirror() {
    if (currentFacingMode === 'user') {
        videoPreview.classList.add('mirrored');
    } else {
        videoPreview.classList.remove('mirrored');
    }
}

function initSendCanvas() {
    cleanupSendCanvas();
    sendCanvas = document.createElement('canvas');
    sendCanvas.width = SEND_WIDTH;
    sendCanvas.height = SEND_HEIGHT;
    sendCanvasCtx = sendCanvas.getContext('2d');
    sendStream = sendCanvas.captureStream(VIDEO_FPS);
    canvasDrawInterval = setInterval(() => {
        if (videoPreview.readyState >= 2 && sendCanvasCtx) {
            sendCanvasCtx.drawImage(videoPreview, 0, 0, SEND_WIDTH, SEND_HEIGHT);
        }
    }, 1000 / VIDEO_FPS);
}

function cleanupSendCanvas() {
    if (canvasDrawInterval) { clearInterval(canvasDrawInterval); canvasDrawInterval = null; }
    if (sendStream) { sendStream.getTracks().forEach(t => t.stop()); sendStream = null; }
    sendCanvas = null;
    sendCanvasCtx = null;
}

// 更新切换按钮状态
function updateToggleButton() {
    const toggleBtn = document.getElementById('toggleVideoBtn');
    if (!toggleBtn) return;
    
    const iconEl = toggleBtn.querySelector('.ctrl-icon-fa');
    const textEl = toggleBtn.querySelector('span:last-child');
    
    if (isVideoStreaming) {
        toggleBtn.classList.add('active');
        if (iconEl) { iconEl.classList.remove('fa-video'); iconEl.classList.add('fa-video-slash'); }
        if (textEl) textEl.textContent = 'Stop';
    } else {
        toggleBtn.classList.remove('active');
        if (iconEl) { iconEl.classList.remove('fa-video-slash'); iconEl.classList.add('fa-video'); }
        if (textEl) textEl.textContent = 'Start';
    }
}

// 开启摄像头
async function startVideo() {
    // 尝试解锁音频自动播放
    unlockAudioContext();
    // 解锁 Web Audio Context (Step 2)
    unlockWebAudioContext();

    try {
        // 先获取会话锁
        try {
            const sessionResponse = await fetch('/api/acquire_session', { method: 'POST' });
            const sessionResult = await sessionResponse.json();
            
            if (!sessionResult.success) {
                log(`⚠️ ${sessionResult.error}`, 'error');
                
                // 询问用户是否要强制释放会话
                const shouldRelease = confirm('系统正在被其他用户使用中。\n\n是否强制释放会话？\n（如果上一个用户已经离开，可以安全释放）');
                
                if (shouldRelease) {
                    // 强制释放（不显示弹窗）
                    await forceReleaseSession(false);
                    // 重新尝试获取会话
                    const retryResponse = await fetch('/api/acquire_session', { method: 'POST' });
                    const retryResult = await retryResponse.json();
                    
                    if (!retryResult.success) {
                        log(`⚠️ 重试失败: ${retryResult.error}`, 'error');
                        alert('获取会话失败，请稍后再试');
                        return;
                    }
                    
                    currentSessionId = retryResult.session_id;
                    log(`✅ 会话已获取`, 'success');
                } else {
                    return;
                }
            } else {
                currentSessionId = sessionResult.session_id;
                log(`✅ 会话已获取`, 'success');
            }
        } catch (e) {
            log(`获取会话锁失败: ${e.message}`, 'error');
            return;
        }
        
        log('正在请求摄像头权限...', 'info');
        
        // 通知服务端清理文件夹
        try {
            const response = await fetch('/api/start_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            });
            const result = await response.json();
            if (result.success) {
                log('已通知服务端清理文件夹', 'info');
            }
        } catch (e) {
            log(`通知服务端失败: ${e.message}`, 'warning');
        }
        
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: currentFacingMode,
                frameRate: { ideal: VIDEO_FPS }
            },
            audio: false
        });
        
        videoPreview.srcObject = videoStream;
        updateVideoMirror();
        initSendCanvas();
        isVideoStreaming = true;
        
        // 更新切换按钮状态
        updateToggleButton();
        liveStatus.style.display = 'flex';
        
        const vt = videoStream.getVideoTracks()[0];
        const settings = vt ? vt.getSettings() : {};
        log(`摄像头已开启 (${settings.width||'?'}x${settings.height||'?'})，发送 ${SEND_WIDTH}x${SEND_HEIGHT}`, 'success');
        
        // 开始每秒捕获并发送视频帧
        startFrameCapture();
        
    } catch (error) {
        log(`摄像头开启失败: ${error.message}`, 'error');
    }
}

// 停止视频
async function stopVideo() {
    // 立即标记为停止，阻止后续的递归调用
    isVideoStreaming = false;
    
    // 清理视频录制相关资源
    if (videoRecordTimeout) {
        clearTimeout(videoRecordTimeout);
        videoRecordTimeout = null;
    }
    
    if (videoMediaRecorder && videoMediaRecorder.state === 'recording') {
        try {
            videoMediaRecorder.stop();
            // 显式清理录制器引用
            videoMediaRecorder.ondataavailable = null;
            videoMediaRecorder.onstop = null;
            videoMediaRecorder.onerror = null;
        } catch (e) {
            console.error("Error stopping recorder:", e);
        }
    }
    videoMediaRecorder = null;
    
    cleanupSendCanvas();
    
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    
    // 清理所有轮询定时器
    if (pollInterval) {
        clearTimeout(pollInterval);
        pollInterval = null;
    }

    // 停止 SSE 流 (token + TTS chunk + error)
    stopEventStream();

    // 停止 TTS 音频
    stopAllWebAudioPlayback(true);

    videoPreview.srcObject = null;
    videoPreview.classList.remove('mirrored');
    
    // 更新切换按钮状态
    updateToggleButton();
    liveStatus.style.display = 'none';
    if (sendingStatus) sendingStatus.style.display = 'none';
    
    // 清理服务器端录制的音视频文件
    try {
        const response = await fetch('/api/clear_media', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId })
        });
        const result = await response.json();
        if (result.success) {
            log(`已清理媒体文件`, 'success');
        }
    } catch (error) {
        log(`清理媒体文件失败: ${error.message}`, 'warning');
    }
    
    // 重置统计
    frameCount = 0;
    totalDataSize = 0;
    updateStats();
    document.getElementById('fps').textContent = '0';
    
    // 隐藏响应面板
    responsePanel.classList.remove('visible');

    // 清空浮动响应
    const floatingContainer = document.getElementById('floatingResponses');
    if (floatingContainer) {
        floatingContainer.innerHTML = '';
    }

    // 重置流式显示状态（防止内存泄漏）
    currentStreamingBubble = null;
    currentStreamingResponseId = null;
    currentDisplayedResponseId = null;
    streamedResponseIds.clear();
    
    log('视频流已停止', 'warning');
    
    // 自动释放会话（释放按钮已移除，停止时自动释放）
    await forceReleaseSession(false);
}

// 强制释放会话（当会话被卡住时使用）
// showAlert: 是否显示弹窗提示（手动点击按钮时显示，自动调用时不显示）

async function switchCamera() {
    // 切换模式
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    const modeName = currentFacingMode === 'user' ? '前置' : '后置';
    log(`切换至${modeName}摄像头`, 'info');
    
    // 如果当前正在直播，则重新启动流
    if (isVideoStreaming) {
        // 停止当前流（但不重置UI状态，只为了重启）
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
        }
        
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: currentFacingMode,
                    frameRate: { ideal: VIDEO_FPS }
                },
                audio: false
            });
            
            videoPreview.srcObject = videoStream;
            updateVideoMirror();
            initSendCanvas();
            log(`已切换到${modeName}摄像头`, 'success');
            
        } catch (error) {
            log(`切换摄像头失败: ${error.message}`, 'error');
            stopVideo(); // 出错则完全停止
        }
    } else {
        log(`下次启动将使用${modeName}摄像头`, 'info');
    }
}

// 开始帧捕获 - 使用 MediaRecorder 录制视频
function startFrameCapture() {
    log(`开始视频捕获: ${VIDEO_FPS}fps, 每 ${VIDEO_SEND_INTERVAL} 秒发送一次`, 'info');
    
    // NOTE: 完整响应轮询已移除，所有内容通过流式 token 接收
    // startPollingResponse();  // REMOVED - using streaming tokens only
    
    // 启动 SSE 连接 (token + chunk + error 三合一)
    startEventStream();

    // 再次尝试解锁音频（以防之前失败）
    unlockAudioContext();

    // 解锁 Web Audio Context (Step 2)
    unlockWebAudioContext();
    
    // 使用 MediaRecorder 录制原始视频流
    function recordAndSend() {
        if (!isVideoStreaming || !videoStream) return;
        
        try {
            let chunks = [];
            
            // 检测浏览器支持的视频格式（Safari 不支持 WebM）
            const videoMimeTypes = [
                'video/webm;codecs=vp8',     // Chrome, Firefox
                'video/webm;codecs=vp9',     // Chrome
                'video/webm',                // 通用 WebM
                'video/mp4;codecs=h264',     // Safari (iOS 14.3+)
                'video/mp4',                 // Safari 回退
            ];
            
            let selectedMimeType = '';
            for (const mimeType of videoMimeTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    selectedMimeType = mimeType;
                    log(`✓ 使用视频格式: ${mimeType}`, 'info');
                    break;
                }
            }
            
            if (!selectedMimeType) {
                log('⚠️ 浏览器不支持任何已知的视频格式，尝试默认格式', 'warning');
            }
            
            // 创建新的 recorder 实例
            const recorderOptions = selectedMimeType ? {
                mimeType: selectedMimeType,
                videoBitsPerSecond: 500000  // 500 kbps
            } : {
                videoBitsPerSecond: 500000  // 500 kbps（不指定 mimeType，让浏览器选择）
            };
            
            videoMediaRecorder = new MediaRecorder(sendStream || videoStream, recorderOptions);
            
            // 记录实际使用的 MIME 类型
            log(`📹 MediaRecorder 实际格式: ${videoMediaRecorder.mimeType}`, 'info');
        
            videoMediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    chunks.push(event.data);
                }
            };
            
            videoMediaRecorder.onstop = async () => {
                // 及时清理 recorder 引用，防止内存泄漏
                const recorder = videoMediaRecorder;
                if (recorder) {
                    recorder.ondataavailable = null;
                    recorder.onstop = null;
                    recorder.onerror = null;
                }

                // 如果已经停止流传输，不再发送数据
                if (!isVideoStreaming) {
                    chunks = [];
                    return;
                }
                
                if (chunks.length > 0) {
                    // 使用实际录制时的 MIME 类型
                    const actualMimeType = videoMediaRecorder.mimeType || 'video/webm';
                    const videoBlob = new Blob(chunks, { type: actualMimeType });
                    chunks = []; // 立即释放 chunks 数组
                    await sendVideoFrame(videoBlob);
                }
                
                // 继续下一次录制
                if (isVideoStreaming) {
                    // 使用 setTimeout 确保异步执行，避免递归过深
                    setTimeout(recordAndSend, 0);
                }
            };
            
            videoMediaRecorder.onerror = (e) => {
                console.error("MediaRecorder error:", e);
                log(`❌ 视频录制错误: ${e.error ? e.error.name : '未知错误'}`, 'error');
                if (isVideoStreaming) {
                    setTimeout(recordAndSend, 1000);
                }
            };

            // 开始录制
            videoMediaRecorder.start();
            
            // 录制指定时长后停止
            videoRecordTimeout = setTimeout(() => {
                if (videoMediaRecorder && videoMediaRecorder.state === 'recording') {
                    videoMediaRecorder.stop();
                }
            }, VIDEO_SEND_INTERVAL * 1000);
            
        } catch (e) {
            console.error("Video recording error:", e);
            log(`❌ 视频录制初始化失败: ${e.message}`, 'error');
            
            // Safari 特殊处理：如果 MediaRecorder 完全不可用
            if (e.name === 'NotSupportedError') {
                log('⚠️ 您的浏览器版本过旧，不支持视频录制功能。请升级 Safari 到 14.3+ 版本。', 'error');
                stopVideo();
                alert('您的浏览器不支持视频录制功能。\n\niOS Safari 需要 14.3 或更高版本。\n请升级系统后重试。');
                return;
            }
            
            // 出错后尝试恢复
            if (isVideoStreaming) {
                setTimeout(recordAndSend, 1000);
            }
        }
    }
    
    // 开始第一次录制
    recordAndSend();
}

// 发送视频帧
async function sendVideoFrame(blob) {
    try {
        if (sendingStatus) sendingStatus.style.display = 'flex';
        
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        formData.append('session_id', currentSessionId || '');  // 携带 session_id
        
        const response = await fetch('/api/video', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // 更新帧数统计
            const framesPerBatch = Math.round(VIDEO_SEND_INTERVAL * VIDEO_FPS);
            frameCount += framesPerBatch;
            
            totalDataSize += blob.size / 1024;
            updateStats();
            
            // 更新 FPS 显示
            document.getElementById('fps').textContent = VIDEO_FPS.toString();
            
            // 更新服务端连接状态指示
            if (result.sent) {
                // 视频成功发送到服务端
                if (!window.serverConnected) {
                    window.serverConnected = true;
                    log('✓ 已连接到服务端，视频流正在发送', 'success');
                }
            }
        }
        
        setTimeout(() => {
            if (sendingStatus) sendingStatus.style.display = 'none';
        }, 200);
        
    } catch (error) {
        log(`发送视频帧失败: ${error.message}`, 'error');
        if (sendingStatus) sendingStatus.style.display = 'none';
    }
}

