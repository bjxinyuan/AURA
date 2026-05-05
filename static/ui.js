// aura frontend — logging and status utilities.

// 日志函数（限制最大条目数，防止内存泄漏）
var MAX_LOG_ENTRIES = 100;  // 最多保留100条日志
function log(message, type = 'info') {
    const logContent = document.getElementById('logContent');
    if (!logContent) return;
    
    const time = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="time">[${time}]</span>${message}`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
    
    // 限制日志条目数量，防止内存泄漏
    while (logContent.children.length > MAX_LOG_ENTRIES) {
        logContent.removeChild(logContent.firstChild);
    }
}

// 更新统计
function updateStats() {
    document.getElementById('frameCount').textContent = frameCount;
    document.getElementById('dataSize').textContent = 
        totalDataSize > 1024 ? `${(totalDataSize / 1024).toFixed(1)} MB` : `${totalDataSize.toFixed(1)} KB`;
}

// 切换视频开启/关闭
