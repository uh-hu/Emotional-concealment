document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const micBtn = document.getElementById('micBtn');
    const fileInput = document.getElementById('fileInput');
    const fileStatus = document.getElementById('fileStatus');
    const fileName = document.getElementById('fileName');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const submitBtn = document.getElementById('submitBtn');
    const uploadArea = document.getElementById('uploadArea');
    
    // Panels
    const inputPanel = document.getElementById('inputPanel');
    const loadingPanel = document.getElementById('loadingPanel');
    const resultPanel = document.getElementById('resultPanel');
    
    // Loading components
    const loadingText = document.getElementById('loadingText');
    const progressBar = document.getElementById('progressBar');
    const resetBtn = document.getElementById('resetBtn');

    // State
    let isRecording = false;
    let audioFile = null;
    let recordingTimeout = null;

    // ----- File Upload & Drag-Drop -----
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (isRecording) stopRecording();

        if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
            audioFile = file;
            fileName.textContent = file.name;
            fileStatus.classList.remove('hidden');
            submitBtn.disabled = false;
        } else {
            alert('请上传有效的音频或视频文件。');
        }
    }

    removeFileBtn.addEventListener('click', () => {
        audioFile = null;
        fileInput.value = '';
        fileStatus.classList.add('hidden');
        submitBtn.disabled = true;
    });

    // ----- Microphone Recording (Simulation/Real) -----
    
    micBtn.addEventListener('click', async () => {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    function startRecording() {
        isRecording = true;
        micBtn.classList.add('recording');
        micBtn.innerHTML = '<i class="fas fa-stop"></i>';
        
        if (audioFile) {
            removeFileBtn.click();
        }

        submitBtn.disabled = true;
        submitBtn.innerHTML = '正在录音... <i class="fas fa-microphone-lines"></i>';

        recordingTimeout = setTimeout(() => {
            if (isRecording) stopRecording();
        }, 5000); // 录制5秒后自动停止
    }

    function stopRecording() {
        isRecording = false;
        clearTimeout(recordingTimeout);
        
        micBtn.classList.remove('recording');
        micBtn.innerHTML = '<i class="fas fa-microphone"></i>';
        
        fileName.textContent = `语料录制_${new Date().getTime().toString().slice(-6)}.wav`;
        fileStatus.classList.remove('hidden');
        audioFile = { name: fileName.textContent }; // Mock file
        
        submitBtn.disabled = false;
        submitBtn.innerHTML = '开始深度分析 <i class="fas fa-arrow-right"></i>';
    }

    // ----- Analysis Process -----
    
    submitBtn.addEventListener('click', () => {
        if (!audioFile) return;

        inputPanel.classList.remove('active');
        setTimeout(() => {
            inputPanel.style.display = 'none';
            loadingPanel.style.display = 'block';
            
            requestAnimationFrame(() => {
                loadingPanel.classList.add('active');
            });
            
            startLoadingSequence();
        }, 400); 
    });

    function startLoadingSequence() {
        const steps = [
            { text: "提取声学特征 (Prosody Extraction)...", progress: 20, time: 600 },
            { text: "语音转文本对齐 (Speech-to-Text Alignment)...", progress: 45, time: 1400 },
            { text: "语义深度解析 (Semantic Comprehension)...", progress: 70, time: 2200 },
            { text: "生成模态偏移计算图谱...", progress: 90, time: 3000 },
            { text: "完成极化特征量化评估...", progress: 100, time: 3800 }
        ];

        steps.forEach(step => {
            setTimeout(() => {
                loadingText.textContent = step.text;
                progressBar.style.width = step.progress + '%';
                
                if (step.progress === 100) {
                    setTimeout(showResult, 500); 
                }
            }, step.time);
        });
    }

    // ----- Result Presentation -----
    
    function showResult() {
        loadingPanel.classList.remove('active');
        
        setTimeout(() => {
            loadingPanel.style.display = 'none';
            resultPanel.style.display = 'block';
            
            progressBar.style.width = '0%'; 
            
            requestAnimationFrame(() => {
                resultPanel.classList.add('active');
            });
        }, 400);
    }

    resetBtn.addEventListener('click', () => {
        resultPanel.classList.remove('active');
        
        setTimeout(() => {
            resultPanel.style.display = 'none';
            inputPanel.style.display = 'block';
            
            removeFileBtn.click();
            submitBtn.innerHTML = '开始深度分析 <i class="fas fa-arrow-right"></i>';
            
            requestAnimationFrame(() => {
                inputPanel.classList.add('active');
            });
        }, 400);
    });
});
