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

});
