// aura frontend — session acquire/release RPCs.

async function forceReleaseSession(showAlert = true) {
    try {
        log('🔓 正在释放会话...', 'info');

        // 停止所有正在进行的播放和 SSE 流
        stopAllWebAudioPlayback(true);
        stopEventStream();
        
        const response = await fetch('/api/force_release_session', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            log(`✅ ${result.message}`, 'success');
            currentSessionId = null;  // 清空本地会话 ID
            if (showAlert) {
                alert('会话已释放，现在可以重新开启摄像头');
            }
        } else {
            log(`⚠️ 释放失败: ${result.error}`, 'warning');
        }
    } catch (e) {
        log(`❌ 释放会话出错: ${e.message}`, 'error');
    }
}

// 切换摄像头
