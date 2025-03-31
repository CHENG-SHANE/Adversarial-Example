document.addEventListener('DOMContentLoaded', () => {
    const uploadSection = document.querySelector('.upload-section');
    const strengthSliderSection = document.querySelector('.strength-slider-section');
    const uploadBtn = document.querySelector('.upload-btn');
    const fileInput = document.getElementById('file-upload');
    const previewContainer = document.getElementById('preview-container');
    const strengthSlider = document.getElementById('strength-slider');
    const strengthLabel = document.getElementById('strength-label');
    const qrCodeSection = document.getElementById('qr-code-section');
    const qrCodeContainer = document.getElementById('qr-code-container');

    const API_BASE_URL = 'http://localhost:5000'; // 後端 URL

    uploadBtn.addEventListener('click', () => {
        uploadBtn.classList.add('click-animation');
        setTimeout(() => uploadBtn.classList.remove('click-animation'), 300);
    });

    fileInput.addEventListener('change', handleFileChange);

    strengthSlider.addEventListener('input', () => {
        const value = parseFloat(strengthSlider.value).toFixed(3);
        strengthLabel.textContent = value;
        console.log("Selected Strength:", value);
    });

    function handleFileChange() {
        previewContainer.innerHTML = ''; // 清除舊預覽
        qrCodeSection.style.display = 'none'; // 隱藏 QR Code 區域

        const files = fileInput.files; // 獲取多個文件
        if (!files || files.length === 0) {
            alert('Please upload at least one valid image file.');
            return;
        }

        // 預覽每個選中的圖片
        for (const file of files) {
            if (!file.type.startsWith('image/')) {
                alert(`File ${file.name} is not a valid image.`);
                continue;
            }
            const reader = new FileReader();
            reader.onload = (e) => {
                const preview = document.createElement('div');
                preview.classList.add('preview', 'fade-in');
                const img = document.createElement('img');
                img.src = e.target.result;
                preview.appendChild(img);
                previewContainer.appendChild(preview);
            };
            reader.readAsDataURL(file);
        }

        addConfirmButton();
    }
    
    // 添加確認按鈕
    function addConfirmButton() {
        if (!document.getElementById('confirm-btn')) {
            const confirmBtn = document.createElement('button');
            confirmBtn.classList.add('confirm-btn', 'stylish-btn');
            confirmBtn.id = 'confirm-btn';
            confirmBtn.textContent = 'Confirm Encryption';
            confirmBtn.onclick = confirmEncryption;
            strengthSliderSection.appendChild(confirmBtn);
        }
    }
    
    async function confirmEncryption() {
        const files = fileInput.files;
        if (!files || files.length === 0) {
            alert("Please upload files before confirming encryption.");
            return;
        }
    
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        formData.append('strength', strengthSlider.value);
    
        try {
            const response = await fetch(`${API_BASE_URL}/upload/`, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorText = await response.text();
                console.error("Server error:", errorText);
                throw new Error(`Server error: ${response.statusText}`);
            }
    
            const responseJson = await response.json();
            console.log("Response JSON:", responseJson);
    
            const { processed_files, confidences } = responseJson;
            if (!processed_files || processed_files.length === 0) {
                throw new Error('No processed files returned from server.');
            }
    
            // 明確將 confidences 傳入
            displayProcessedFiles(processed_files, confidences);
    
        } catch (error) {
            console.error("Encryption error:", error);
            alert("An error occurred during encryption. Please check your connection and try again.");
        }
    }
    
    function displayProcessedFiles(filenames, confidences) {
        uploadSection.style.display = 'none';
        strengthSliderSection.style.display = 'none';
    
        previewContainer.innerHTML = '';
        qrCodeContainer.innerHTML = '';
    
        qrCodeSection.style.display = 'block';
    
        filenames.forEach((filename, index) => {
            const fileUrl = `${API_BASE_URL}/download/${encodeURIComponent(filename)}`;
            qrCodeContainer.appendChild(document.createElement('br'));
            // 顯示信心值
            const confidenceText = document.createElement('p');
            confidenceText.textContent = `FaceNet Confidence: ${(confidences[index] * 100).toFixed(2)}%`;
            confidenceText.style.fontWeight = 'bold';
            confidenceText.style.color = '#FF7744';
            qrCodeContainer.appendChild(confidenceText);

            // 生成 QR Code
            const qrCanvas = document.createElement('canvas');
            qrCodeContainer.appendChild(qrCanvas);
    
            QRCode.toCanvas(qrCanvas, fileUrl, (error) => {
                if (error) {
                    console.error('QR Code generation failed:', fileUrl, error);
                    alert(`Failed to generate QR Code for: ${fileUrl}`);
                }
            });
    
            // 加入下載按鈕
            const downloadBtn = document.createElement('a');
            downloadBtn.textContent = `Download ${filename}`;
            downloadBtn.href = fileUrl;
            downloadBtn.download = filename;
            downloadBtn.classList.add('stylish-btn');
            qrCodeContainer.appendChild(downloadBtn);

    
            qrCodeContainer.appendChild(document.createElement('br'));
        });
    }
    
});
