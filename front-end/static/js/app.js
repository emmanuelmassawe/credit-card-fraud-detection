// ==================== CONFIG ====================
const API_URL = window.location.origin;  // locally and also in docker

// ==================== SAMPLE DATA ====================
const sampleTransactions = {
    normal: {
        Time: 0, Amount: 149.62,
        V1: -1.3598071336738, V2: -0.0727811733098497, V3: 2.53634673796914,
        V4: 1.37815522427443, V5: -0.338320769942518, V6: 0.462387777762292,
        V7: 0.239598554061257, V8: 0.0986979012610507, V9: 0.363786969611213,
        V10: 0.0907941719789316, V11: -0.551599533260813, V12: -0.617800855762348,
        V13: -0.991389847235408, V14: -0.311169353699879, V15: 1.46817697209427,
        V16: -0.470400525259478, V17: 0.207971241929242, V18: 0.0257905801985591,
        V19: 0.403992960255733, V20: 0.251412098239705, V21: -0.018306777944153,
        V22: 0.277837575558899, V23: -0.110473910188767, V24: 0.0669280749146731,
        V25: 0.128539358273528, V26: -0.189114843888824, V27: 0.133558376740387,
        V28: -0.0210530534538215
    },
    fraud: {
        Time: 406, Amount: 0.00,
        V1: -2.3122265423263, V2: 1.95199201064158, V3: -1.60985073229769,
        V4: 3.9979055875468, V5: -0.522187864667764, V6: -1.42654531920595,
        V7: -2.53738730624579, V8: 1.39165724829804, V9: -2.77008927719433,
        V10: -2.77227214465915, V11: 3.20203320709635, V12: -2.89990738849473,
        V13: -0.595221881324605, V14: -4.28925378244217, V15: 0.389724120274487,
        V16: -1.14074717980657, V17: -2.83005567450437, V18: -0.0168224681808257,
        V19: 0.416955705037907, V20: 0.126910559061474, V21: 0.517232370861764,
        V22: -0.0350493686052974, V23: -0.465211076182388, V24: 0.320198198514526,
        V25: 0.0445191674731724, V26: 0.177839798284401, V27: 0.261145002567677,
        V28: -0.143275874698919
    }
};

// ==================== LOAD TRANSACTION ====================
function loadTransaction(data) {
    document.getElementById('time').value = data.Time;
    document.getElementById('amount').value = data.Amount;
    for (let i = 1; i <= 28; i++) {
        document.getElementById(`v${i}`).value = data[`V${i}`];
    }
    document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
}

function loadNormalTransaction() {
    loadTransaction(sampleTransactions.normal);
    showToast('Normal transaction loaded!', 'success');
}

function loadFraudTransaction() {
    loadTransaction(sampleTransactions.fraud);
    showToast('Fraud transaction loaded!', 'danger');
}

function loadRandomTransaction() {
    const data = {
        Time: Math.floor(Math.random() * 10000),
        Amount: parseFloat((Math.random() * 500).toFixed(2))
    };
    for (let i = 1; i <= 28; i++) {
        data[`V${i}`] = parseFloat(((Math.random() - 0.5) * 6).toFixed(6));
    }
    loadTransaction(data);
    showToast('Random transaction loaded!', 'info');
}

// ==================== TOAST NOTIFICATION ====================
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 80px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        color: white;
        font-weight: 600;
        z-index: 9999;
        animation: slideInRight 0.3s ease;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.2);
        min-width: 250px;
    `;

    const colors = {
        success: 'linear-gradient(135deg, #10b981, #059669)',
        danger:  'linear-gradient(135deg, #ef4444, #dc2626)',
        info:    'linear-gradient(135deg, #667eea, #764ba2)'
    };

    toast.style.background = colors[type] || colors.info;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}

// ==================== FORM SUBMIT ====================
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

    // Collect form data
    const transactionData = {
        Time:   parseFloat(document.getElementById('time').value),
        Amount: parseFloat(document.getElementById('amount').value)
    };
    for (let i = 1; i <= 28; i++) {
        transactionData[`V${i}`] = parseFloat(document.getElementById(`v${i}`).value);
    }

    try {
        // Use API_URL variable to point to FastAPI
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(transactionData)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        showToast('Error: ' + error.message, 'danger');
        console.error(error);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Transaction';
    }
});

// ==================== DISPLAY RESULTS ====================
function displayResults(result) {
    const resultsCard   = document.getElementById('resultsCard');
    const resultContent = document.getElementById('resultContent');
    const emptyState    = document.getElementById('emptyState');

    const isFraud   = result.prediction === 1;
    const fraudPct  = (result.fraud_probability * 100).toFixed(2);
    const confPct   = result.confidence.toFixed(2);

    const riskLevel =
        result.fraud_probability > 0.8 ? '🔴 High Risk' :
        result.fraud_probability > 0.5 ? '🟠 Medium Risk' :
                                         '🟢 Low Risk';

    const progressColor = isFraud
        ? 'linear-gradient(135deg, #ef4444, #dc2626)'
        : 'linear-gradient(135deg, #10b981, #059669)';

    resultContent.innerHTML = `
        <div class="result-verdict ${isFraud ? 'fraud' : 'normal'}">
            <div class="verdict-icon ${isFraud ? 'fraud' : 'normal'}">
                <i class="fas ${isFraud ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
            </div>
            <div class="verdict-text ${isFraud ? 'fraud' : 'normal'}">
                <h2>${result.prediction_label}</h2>
                <p>${isFraud ? '⚠️ Suspicious activity detected' : '✅ Transaction appears legitimate'}</p>
            </div>
        </div>

        <div class="metric-item">
            <div class="metric-label">
                <span>Fraud Probability</span>
                <span class="metric-value">${fraudPct}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill"
                     style="width: ${fraudPct}%; background: ${progressColor};">
                </div>
            </div>
        </div>

        <div class="metric-item">
            <div class="metric-label">
                <span>Model Confidence</span>
                <span class="metric-value">${confPct}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill"
                     style="width: ${confPct}%; background: linear-gradient(135deg, #667eea, #764ba2);">
                </div>
            </div>
        </div>

        <div class="result-grid">
            <div class="result-item">
                <strong>Prediction</strong>
                <span>${result.prediction_label}</span>
            </div>
            <div class="result-item">
                <strong>Risk Level</strong>
                <span>${riskLevel}</span>
            </div>
            <div class="result-item">
                <strong>Model Used</strong>
                <span>XGBoost + SMOTE</span>
            </div>
            <div class="result-item">
                <strong>Action</strong>
                <span>${isFraud ? '⚠️ Review Required' : '✅ Approved'}</span>
            </div>
        </div>
    `;

    emptyState.style.display  = 'none';
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    showToast(
        isFraud ? '⚠️ Fraud Detected!' : '✅ Transaction is Normal',
        isFraud ? 'danger' : 'success'
    );
}

// ==================== NAVIGATION ====================
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) target.scrollIntoView({ behavior: 'smooth' });
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
    });
});

// Toast animation style
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(100px); }
        to   { opacity: 1; transform: translateX(0); }
    }
`;
document.head.appendChild(style);