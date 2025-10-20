// Batch analysis functionality
let batchFiles = [];

document.getElementById('batchFileInput').addEventListener('change', handleBatchFileSelect);
document.getElementById('batchUploadArea').addEventListener('click', () => {
    document.getElementById('batchFileInput').click();
});

// Drag and drop for batch
const batchUploadArea = document.getElementById('batchUploadArea');

batchUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    batchUploadArea.classList.add('dragover');
});

batchUploadArea.addEventListener('dragleave', () => {
    batchUploadArea.classList.remove('dragover');
});

batchUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    batchUploadArea.classList.remove('dragover');
    handleBatchFileSelect({ target: { files: e.dataTransfer.files } });
});

function handleBatchFileSelect(event) {
    const files = Array.from(event.target.files);
    
    // Filter valid image files
    const validFiles = files.filter(file => file.type.match('image.*'));
    
    if (validFiles.length === 0) {
        showAlert('Please select valid image files (JPG, PNG, JPEG)', 'warning');
        return;
    }
    
    batchFiles = validFiles;
    displayFileList();
}

function displayFileList() {
    const fileList = document.getElementById('fileList');
    const fileListContainer = document.getElementById('fileListContainer');
    
    fileListContainer.innerHTML = '';
    
    batchFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'list-group-item d-flex justify-content-between align-items-center';
        fileItem.innerHTML = `
            <div>
                <i class="fas fa-file-image text-primary me-2"></i>
                ${file.name}
                <small class="text-muted">(${(file.size / 1024 / 1024).toFixed(2)} MB)</small>
            </div>
            <button class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileListContainer.appendChild(fileItem);
    });
    
    fileList.style.display = 'block';
    batchUploadArea.style.display = 'none';
}

function removeFile(index) {
    batchFiles.splice(index, 1);
    if (batchFiles.length === 0) {
        clearBatchFiles();
    } else {
        displayFileList();
    }
}

function clearBatchFiles() {
    batchFiles = [];
    document.getElementById('batchFileInput').value = '';
    document.getElementById('fileList').style.display = 'none';
    document.getElementById('batchUploadArea').style.display = 'block';
    document.getElementById('batchResults').style.display = 'none';
}

async function analyzeBatch() {
    if (batchFiles.length === 0) {
        showAlert('Please select some images first', 'warning');
        return;
    }
    
    const analyzeBtn = document.getElementById('analyzeBatchBtn');
    const progressSection = document.getElementById('batchProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultsSection = document.getElementById('batchResults');
    
    analyzeBtn.disabled = true;
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    const formData = new FormData();
    batchFiles.forEach(file => formData.append('files', file));
    
    try {
        const response = await fetch('/analyze_batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayBatchResults(data);
        } else {
            showAlert('Error analyzing batch: ' + data.error, 'danger');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'danger');
    } finally {
        analyzeBtn.disabled = false;
        progressSection.style.display = 'none';
    }
}

function displayBatchResults(data) {
    const resultsSection = document.getElementById('batchResults');
    
    let resultsHTML = `
        <div class="alert alert-success">
            <h4><i class="fas fa-check-circle me-2"></i>Batch Analysis Complete</h4>
            <p>Processed ${data.total_processed} images with ${data.total_errors} errors</p>
        </div>
    `;
    
    if (data.errors.length > 0) {
        resultsHTML += `
            <div class="alert alert-warning">
                <h5>Errors:</h5>
                <ul>
                    ${data.errors.map(error => `<li>${error}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    resultsHTML += '<div class="row">';
    
    data.results.forEach((result, index) => {
        resultsHTML += `
            <div class="col-md-6 mb-3">
                <div class="card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <strong>${result.filename}</strong>
                        <span class="badge ${result.prediction === 'NORMAL' ? 'bg-success' : 'bg-danger'}">
                            ${result.prediction}
                        </span>
                    </div>
                    <div class="card-body">
                        <div class="text-center mb-3">
                            <img src="${result.image}" class="img-fluid rounded" style="max-height: 150px;">
                        </div>
                        <div class="mb-2">
                            <strong>Confidence:</strong> ${result.confidence}%
                        </div>
                        <div class="mb-2">
                            <strong>Risk Level:</strong>
                            <span class="badge ${result.risk_level === 'HIGH' ? 'bg-danger' : result.risk_level === 'MEDIUM' ? 'bg-warning' : 'bg-success'}">
                                ${result.risk_level}
                            </span>
                        </div>
                        <div class="progress mb-2" style="height: 10px;">
                            <div class="progress-bar bg-success" style="width: ${result.normal_probability}%"></div>
                            <div class="progress-bar bg-danger" style="width: ${result.pneumonia_probability}%"></div>
                        </div>
                        <small class="text-muted">Normal: ${result.normal_probability}% | Pneumonia: ${result.pneumonia_probability}%</small>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultsHTML += '</div>';
    
    resultsSection.innerHTML = resultsHTML;
    resultsSection.style.display = 'block';
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('.card-body').insertBefore(alertDiv, document.querySelector('.card-body').firstChild);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}