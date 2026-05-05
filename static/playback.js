// aura frontend — Web Audio API streaming PCM playback.

// =============== Web Audio API 流式 PCM 播放 (Step 2) ===============

// 初始化 Web Audio API
function initWebAudioContext() {
    if (webAudioContext) return;
    
    try {
        webAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        webAudioNextPlayTime = webAudioContext.currentTime;
        log('🔊 Web Audio API 已初始化 (流式 PCM 播放)', 'success');
    } catch (e) {
        log(`🔊 Web Audio API 初始化失败: ${e.message}`, 'error');
    }
}

// 解锁 Web Audio Context (需要用户交互)
function unlockWebAudioContext() {
    if (!webAudioContext) {
        initWebAudioContext();
    }
    if (webAudioContext && webAudioContext.state === 'suspended') {
        webAudioContext.resume().then(() => {
            log('🔊 Web Audio Context 已解锁', 'info');
        });
    }
}

// 停止所有当前播放的音频 (用于新 response 到达时中断旧播放)
// resetResponseId: 是否重置 response_id (录音中断时需要重置，新响应切换时不需要)
function stopAllWebAudioPlayback(resetResponseId = false) {
    for (const source of webAudioActiveSources) {
        try {
            source.stop();
        } catch (e) {
            // 忽略已经停止的 source
        }
    }
    webAudioActiveSources = [];
    webAudioChunkQueue = [];
    webAudioIsPlaying = false;
    
    // 重置 response_id，防止后续 chunk 继续播放
    if (resetResponseId) {
        if (webAudioCurrentResponseId) {
            canceledResponseIds.add(webAudioCurrentResponseId);
            log(`🚫 已取消 WebAudio 响应: ${webAudioCurrentResponseId}`, 'warning');
        }
        webAudioCurrentResponseId = null;
        webAudioNextPlayTime = 0;
        
        // 隐藏 TTS 指示器
        const floatingTts = document.getElementById('floatingTtsIndicator');
        if (floatingTts) floatingTts.classList.remove('visible');
    }
}

// 播放 PCM chunk (int16 -> float32 -> AudioBuffer)
function playPcmChunk(pcmBase64, sampleRate, responseId, sentenceIdx, chunkIdx, isFinal) {
    // 检查是否是被取消的响应
    if (canceledResponseIds.has(responseId)) {
        return;
    }
    
    // 录音期间不播放任何 TTS
    if (isAudioRecording) {
        return;
    }
    
    if (!webAudioContext) {
        initWebAudioContext();
    }
    if (!webAudioContext) return;
    
    // 检查是否是新的响应，如果是则立即中断当前播放
    if (webAudioCurrentResponseId !== responseId) {
        // 新响应开始 - 立即停止所有当前播放
        if (webAudioActiveSources.length > 0) {
            log(`⏹ [WebAudio] 中断旧响应播放，切换到新响应`, 'info');
            stopAllWebAudioPlayback();
        }
        
        // 再次检查新 responseId 是否在取消列表中（双重保险）
        if (canceledResponseIds.has(responseId)) {
            return;
        }
        
        webAudioCurrentResponseId = responseId;
        webAudioCurrentSentenceIdx = sentenceIdx;
        webAudioNextPlayTime = webAudioContext.currentTime;  // 从当前时间开始播放
        webAudioIsPlaying = true;
        
        // 显示 TTS 指示器
        const floatingTts = document.getElementById('floatingTtsIndicator');
        if (floatingTts) floatingTts.classList.add('visible');
        
        log(`🔊 [WebAudio] 开始播放响应: ${responseId.substring(0, 8)}...`, 'info');
    }
    
    // 如果是空数据 (final marker)，不播放
    if (!pcmBase64 || pcmBase64.length === 0) {
        if (isFinal) {
            log(`🔊 [WebAudio] 句子 ${sentenceIdx} 完成`, 'success');
        }
        return;
    }
    
    try {
        // Base64 解码
        const binaryString = atob(pcmBase64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        // Int16 PCM -> Float32
        const int16Array = new Int16Array(bytes.buffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;  // Normalize to [-1, 1]
        }
        
        // 创建 AudioBuffer
        const audioBuffer = webAudioContext.createBuffer(1, float32Array.length, sampleRate);
        audioBuffer.copyToChannel(float32Array, 0);
        
        // 创建 BufferSource 并播放
        const source = webAudioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(webAudioContext.destination);
        
        // 调度播放时间
        const startTime = Math.max(webAudioContext.currentTime, webAudioNextPlayTime);
        source.start(startTime);
        
        // 将 source 添加到活跃列表
        webAudioActiveSources.push(source);
        
        // 更新下一个 chunk 的播放时间
        webAudioNextPlayTime = startTime + audioBuffer.duration;
        
        // 首个 chunk 日志
        if (chunkIdx === 0) {
            log(`🚀 [WebAudio] 首个 chunk 开始播放 (句子 ${sentenceIdx})`, 'info');
        }
        
        // 播放结束时的处理
        source.onended = () => {
            // 从活跃列表中移除
            const idx = webAudioActiveSources.indexOf(source);
            if (idx > -1) {
                webAudioActiveSources.splice(idx, 1);
            }
            
            if (isFinal && webAudioActiveSources.length === 0) {
                // 这是最后一个 chunk 且已播放完
                webAudioIsPlaying = false;
                
                // 隐藏 TTS 指示器
                const floatingTts = document.getElementById('floatingTtsIndicator');
                if (floatingTts) floatingTts.classList.remove('visible');
            }
        };
        
    } catch (e) {
        console.error('PCM 播放错误:', e);
    }
}

// =============== Web Audio API 结束 ===============
