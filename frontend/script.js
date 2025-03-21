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

});
