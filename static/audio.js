// aura frontend — microphone capture, audio stream, and mic button events.

async function startAudioRecording() {
    // 尝试解锁音频自动播放
    unlockAudioContext();
    unlockWebAudioContext();

    if (isAudioRecording) return;
    
    // 中断所有正在播放的 TTS 语音
    log('🔇 录音开始，中断 TTS 播放', 'info');
    stopAllWebAudioPlayback(true);  // 停止 Web Audio API 播放，并重置 response_id 防止后续 chunk 继续播放
    
    // 检查浏览器是否支持必要的 API
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('您的浏览器不支持录音功能。请使用最新版本的浏览器。');
        return;
    }
    
    if (typeof MediaRecorder === 'undefined') {
        alert('您的浏览器不支持录音功能。iOS 需要 14.3 或更高版本。');
        return;
    }
    
    try {
        // 检查 audioStream 是否存在且仍然活跃
        const isStreamActive = audioStream && 
            audioStream.getTracks().length > 0 && 
            audioStream.getTracks().every(track => track.readyState === 'live');
        
        if (!isStreamActive) {
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
            }
            audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                }
            });
            log('麦克风权限已获取', 'success');
        }
        
        audioChunks = [];
        
        // 检测支持的音频格式 (iOS 不支持 webm，需要使用 mp4/aac)
        const mimeTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4',
            'audio/mp4;codecs=aac',
            'audio/aac',
            'audio/ogg;codecs=opus',
            'audio/ogg',
        ];
        
        const supportedTypes = mimeTypes.filter(t => MediaRecorder.isTypeSupported(t));
        let audioMimeType = supportedTypes.length > 0 ? supportedTypes[0] : '';
        
        // 创建 MediaRecorder
        try {
            const recorderOptions = audioMimeType ? { mimeType: audioMimeType } : {};
            audioRecorder = new MediaRecorder(audioStream, recorderOptions);
        } catch (recorderError) {
            // 尝试不带选项创建
            audioRecorder = new MediaRecorder(audioStream);
        }
        
        window.currentAudioMimeType = audioRecorder.mimeType || audioMimeType || 'audio/webm';
        
        audioRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        audioRecorder.start(100);
        isAudioRecording = true;
        audioStartTime = Date.now();
        
        // 更新 UI
        micHint.textContent = '正在录音...';
        micHint.classList.add('recording');
        waveform.classList.add('active');
        
        // 开始更新录音时长
        startAudioDurationTimer();
        
        log('开始录音', 'info');
        
    } catch (error) {
        isAudioRecording = false;
        stopAudioDurationTimer();
        log(`录音失败: ${error.message}`, 'error');
        
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            alert('麦克风权限被拒绝。请在浏览器设置中允许麦克风访问。');
        } else if (error.name === 'NotFoundError') {
            alert('未找到麦克风设备。');
        } else {
            alert(`录音出错: ${error.message}`);
        }
    }
}

// 录音时长更新定时器
let audioDurationTimer = null;

// 开始更新录音时长显示
function startAudioDurationTimer() {
    // 先清除旧的定时器
    stopAudioDurationTimer();
    
    audioDurationTimer = setInterval(() => {
        if (!isAudioRecording || !audioStartTime) {
            stopAudioDurationTimer();
            return;
        }
        const duration = Math.floor((Date.now() - audioStartTime) / 1000);
        const durationEl = document.getElementById('audioDuration');
        if (durationEl) {
            durationEl.textContent = `${duration}s`;
        }
    }, 200);  // 每200ms更新一次，减少性能开销
}

// 停止更新录音时长
function stopAudioDurationTimer() {
    if (audioDurationTimer) {
        clearInterval(audioDurationTimer);
        audioDurationTimer = null;
    }
}

// 停止音频录制并发送
async function stopAudioRecording() {
    stopAudioDurationTimer();
    
    if (!isAudioRecording || !audioRecorder) {
        isAudioRecording = false;
        return;
    }
    
    isAudioRecording = false;
    
    // 更新 UI
    if (micHint) {
        micHint.textContent = '按住按钮开始录音';
        micHint.classList.remove('recording');
    }
    if (waveform) waveform.classList.remove('active');
    
    try {
        // 等待录制完成（设置超时防止卡死）
        await Promise.race([
            new Promise(resolve => {
                audioRecorder.onstop = resolve;
                audioRecorder.stop();
            }),
            new Promise((_, reject) => setTimeout(() => reject(new Error('录音停止超时')), 3000))
        ]);
        
        // 创建音频 Blob (使用实际录制时的 mimeType)
        const actualMimeType = window.currentAudioMimeType || 'audio/webm';
        const audioBlob = new Blob(audioChunks, { type: actualMimeType });
        const duration = Math.floor((Date.now() - audioStartTime) / 1000);
        
        log(`录音完成，时长: ${duration}秒，大小: ${(audioBlob.size / 1024).toFixed(1)}KB`, 'success');
        
        // 更新统计
        const audioSizeEl = document.getElementById('audioSize');
        if (audioSizeEl) audioSizeEl.textContent = `${(audioBlob.size / 1024).toFixed(1)} KB`;
        
        // 发送音频
        await sendAudio(audioBlob, actualMimeType);
    } catch (error) {
        log(`录音处理错误: ${error.message}`, 'error');
    } finally {
        // 清理 audioChunks 释放内存
        audioChunks = [];
    }
}

// 发送音频
async function sendAudio(blob, mimeType = 'audio/webm') {
    try {
        log('🎤 正在发送音频到服务端...', 'info');
        
        // 显示等待状态
        showWaitingResponse();
        
        // 根据 mimeType 确定文件扩展名
        let fileExt = 'webm';
        if (mimeType.includes('mp4') || mimeType.includes('aac') || mimeType.includes('m4a')) {
            fileExt = 'm4a';
        } else if (mimeType.includes('ogg')) {
            fileExt = 'ogg';
        } else if (mimeType.includes('wav')) {
            fileExt = 'wav';
        }
        
        const formData = new FormData();
        formData.append('audio', blob, `audio.${fileExt}`);
        formData.append('session_id', currentSessionId || '');  // 携带 session_id
        formData.append('mime_type', mimeType);  // 告诉服务端实际的格式
        
        const response = await fetch('/api/audio', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            audioCount++;
            document.getElementById('audioCount').textContent = audioCount;
            log('✓ 音频发送成功，等待回复...', 'success');
            
            // 轮询已在 startFrameCapture 中启动
            
        } else {
            hideWaitingResponse();
            log(`❌ 音频发送失败: ${result.error}`, 'error');
        }
        
    } catch (error) {
        hideWaitingResponse();
        log(`❌ 音频发送失败: ${error.message}`, 'error');
    }
}


function initMicButton() {
    const micButton = document.getElementById('micBtn');
    if (!micButton) return;
    
    let recordingStartedByPressEvent = false;
    
    micButton.addEventListener('mousedown', function(e) {
        e.preventDefault();
        micButton.classList.add('pressed');
        recordingStartedByPressEvent = true;
        startAudioRecording();
    });
    
    micButton.addEventListener('mouseup', function(e) {
        e.preventDefault();
        micButton.classList.remove('pressed');
        if (recordingStartedByPressEvent) {
            stopAudioRecording();
            recordingStartedByPressEvent = false;
        }
    });
    
    micButton.addEventListener('mouseleave', function(e) {
        micButton.classList.remove('pressed');
        if (isAudioRecording && recordingStartedByPressEvent) {
            stopAudioRecording();
            recordingStartedByPressEvent = false;
        }
    });
    
    micButton.addEventListener('touchstart', function(e) {
        e.preventDefault();
        e.stopPropagation();
        micButton.classList.add('pressed');
        recordingStartedByPressEvent = true;
        startAudioRecording();
    }, { passive: false });
    
    micButton.addEventListener('touchend', function(e) {
        e.preventDefault();
        e.stopPropagation();
        micButton.classList.remove('pressed');
        if (recordingStartedByPressEvent) {
            stopAudioRecording();
            recordingStartedByPressEvent = false;
        }
    }, { passive: false });
    
    micButton.addEventListener('touchcancel', function(e) {
        micButton.classList.remove('pressed');
        if (isAudioRecording) {
            stopAudioRecording();
        }
        recordingStartedByPressEvent = false;
    }, { passive: false });
    
    log('🎙️ 录音按钮已初始化', 'info');
    
    // Flip 按钮按压视觉反馈
    const flipButton = document.getElementById('switchCameraBtn');
    if (flipButton) {
        flipButton.addEventListener('mousedown', function() { flipButton.classList.add('pressed'); });
        flipButton.addEventListener('mouseup', function() { flipButton.classList.remove('pressed'); });
        flipButton.addEventListener('mouseleave', function() { flipButton.classList.remove('pressed'); });
        flipButton.addEventListener('touchstart', function(e) {
            e.preventDefault();
            flipButton.classList.add('pressed');
        }, { passive: false });
        flipButton.addEventListener('touchend', function(e) {
            e.preventDefault();
            flipButton.classList.remove('pressed');
            switchCamera();
        }, { passive: false });
        flipButton.addEventListener('touchcancel', function() {
            flipButton.classList.remove('pressed');
        }, { passive: false });
    }
}

// 页面关闭前清理
window.onbeforeunload = function() {
    stopAllWebAudioPlayback(true);
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
};