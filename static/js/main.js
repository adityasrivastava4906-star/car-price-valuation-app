const API_BASE = window.location.origin;

let config = null;

async function loadConfig() {
    const res = await fetch(`${API_BASE}/api/config`);
    if (!res.ok) throw new Error("Could not load configuration");
    config = await res.json();
    populateForm(config);
    populateHeroStats(config);
    return config;
}

function populateForm(cfg) {
    const brandSelect = document.getElementById("brand");
    brandSelect.innerHTML = cfg.brands
        .map((b) => `<option value="${b.value}">${b.label}</option>`)
        .join("");

    const yearSelect = document.getElementById("year");
    yearSelect.innerHTML = cfg.years
        .map((y) => {
            const age = cfg.currentYear - y;
            const ageLabel = age === 1 ? "1 yr" : `${age} yrs`;
            return `<option value="${y}">${y} (${ageLabel} old)</option>`;
        })
        .join("");

    const defaultYear = cfg.years.find((y) => cfg.currentYear - y === 6) || cfg.years[Math.floor(cfg.years.length / 2)];
    yearSelect.value = defaultYear;

    const fortuner = cfg.brands.find((b) => b.value === "fortuner");
    if (fortuner) brandSelect.value = "fortuner";
}

function populateHeroStats(cfg) {
    const m = cfg.metrics;
    document.getElementById("statR2").textContent = (m.testR2 * 100).toFixed(1);
    document.getElementById("statMae").textContent = m.testMae.toFixed(2);
    document.getElementById("statAcc").textContent = (m.classAccuracy * 100).toFixed(1);
    document.getElementById("statModel").textContent = m.bestRegressor.replace(" Regressor", "");
}

function updateRangeLabel(inputId, labelId, prefix = "", suffix = "", formatNumber = false) {
    const val = document.getElementById(inputId).value;
    let displayVal = val;
    if (formatNumber) displayVal = parseInt(val, 10).toLocaleString("en-IN");
    document.getElementById(labelId).textContent = prefix + displayVal + suffix;
}

function animateValue(el, start, end, duration, suffix = "") {
    const startTime = performance.now();
    const diff = end - start;

    function step(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + diff * eased;
        el.textContent = current.toFixed(2) + suffix;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function getTierClass(tier) {
    if (tier === "Budget") return "tier-budget";
    if (tier === "Premium") return "tier-premium";
    return "tier-mid";
}

function getLiquidity(brand, age) {
    const highDemand = ["city", "verna", "innova", "fortuner", "corolla", "honda"];
    if (age <= 5 && highDemand.includes(brand)) return "High Demand";
    if (age <= 8) return "Moderate";
    return "Lower Turnover";
}

function getFormData() {
    return {
        brand: document.getElementById("brand").value,
        year: parseInt(document.getElementById("year").value, 10),
        presentPrice: parseFloat(document.getElementById("presentPrice").value),
        kmsDriven: parseInt(document.getElementById("kmsDriven").value, 10),
        fuel: document.querySelector('input[name="fuel"]:checked').value,
        transmission: document.querySelector('input[name="transmission"]:checked').value,
        seller: document.querySelector('input[name="seller"]:checked').value,
    };
}

async function calculateValuation() {
    const btn = document.getElementById("btnCalc");
    const errEl = document.getElementById("errorMsg");
    errEl.classList.remove("visible");
    btn.classList.add("calculating");
    btn.disabled = true;

    try {
        const payload = getFormData();
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Prediction failed");

        const priceEl = document.getElementById("outPriceNum");
        const prevPrice = parseFloat(priceEl.dataset.value || "0");
        priceEl.dataset.value = data.predicted_price_lakhs;

        animateValue(
            document.getElementById("outPriceNum"),
            prevPrice || data.predicted_price_lakhs * 0.92,
            data.predicted_price_lakhs,
            800
        );

        const tierEl = document.getElementById("outTier");
        tierEl.textContent = `${data.tier} Tier`;
        tierEl.className = "tier-badge " + getTierClass(data.tier);

        document.getElementById("outRetention").textContent = `${data.retention_pct}%`;
        document.getElementById("outDeprec").textContent = `${data.depreciation_pct_per_year}% / yr`;
        document.getElementById("outLiquidity").textContent = getLiquidity(payload.brand, data.car_age);

        const spectrumPct = Math.min(100, Math.max(8, (data.predicted_price_lakhs / 25) * 100));
        document.getElementById("meterFill").style.width = `${spectrumPct}%`;
    } catch (err) {
        errEl.textContent = err.message;
        errEl.classList.add("visible");
    } finally {
        btn.classList.remove("calculating");
        btn.disabled = false;
    }
}

function initScrollReveal() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );

    document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
}

function initGalleryAnimations() {
    document.querySelectorAll(".gallery-item").forEach((item) => {
        const img = item.querySelector("img");
        if (!img) return;

        const markLoaded = () => item.classList.add("img-loaded");

        if (img.complete && img.naturalWidth > 0) {
            markLoaded();
        } else {
            img.addEventListener("load", markLoaded, { once: true });
        }
    });

    const galleryObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("img-animate");
                    galleryObserver.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.25, rootMargin: "0px 0px -30px 0px" }
    );

    document.querySelectorAll(".gallery-item").forEach((item) => galleryObserver.observe(item));
}

function initLightbox() {
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightboxImg");
    const closeBtn = document.getElementById("lightboxClose");

    document.querySelectorAll(".gallery-item").forEach((item) => {
        item.addEventListener("click", () => {
            const img = item.querySelector("img");
            lightboxImg.src = img.src;
            lightboxImg.alt = img.alt;
            lightbox.classList.add("open");
            document.body.style.overflow = "hidden";
        });
    });

    function close() {
        lightbox.classList.remove("open");
        document.body.style.overflow = "";
    }

    closeBtn.addEventListener("click", close);
    lightbox.addEventListener("click", (e) => {
        if (e.target === lightbox) close();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") close();
    });
}

function bindEvents() {
    document.getElementById("valuationForm").addEventListener("submit", (e) => {
        e.preventDefault();
        calculateValuation();
    });

    document.getElementById("presentPrice").addEventListener("input", () =>
        updateRangeLabel("presentPrice", "presentPriceVal", "₹ ", " Lakhs")
    );
    document.getElementById("kmsDriven").addEventListener("input", () =>
        updateRangeLabel("kmsDriven", "kmsVal", "", " km", true)
    );
}

async function init() {
    bindEvents();
    initScrollReveal();
    initGalleryAnimations();
    initLightbox();

    updateRangeLabel("presentPrice", "presentPriceVal", "₹ ", " Lakhs");
    updateRangeLabel("kmsDriven", "kmsVal", "", " km", true);

    try {
        await loadConfig();
        await calculateValuation();
    } catch (err) {
        document.getElementById("errorMsg").textContent =
            "API unavailable — start the Flask server with: python app.py";
        document.getElementById("errorMsg").classList.add("visible");
    }
}

document.addEventListener("DOMContentLoaded", init);
