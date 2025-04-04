let latestData = null;

function showModal() {
  document.getElementById("validationModal").style.display = "flex";
}

function closeModal() {
  document.getElementById("validationModal").style.display = "none";
}

function updateValue(slider) {
  const valueId = slider.id + "_value";
  document.getElementById(valueId).textContent = slider.value;
}

function predictPrice() {
  const battery_power = parseInt(document.getElementById("battery_power").value);
  const px_height = parseInt(document.getElementById("px_height").value);
  const px_width = parseInt(document.getElementById("px_width").value);
  const ram = parseInt(document.getElementById("ram").value);

  if (isNaN(battery_power) || isNaN(px_height) || isNaN(px_width) || isNaN(ram)) {
    showModal();
    return;
  }

  const isValid =
  battery_power >= 501 && battery_power <= 1998 &&
  px_height >= 0 && px_height <= 1960 &&
  px_width >= 500 && px_width <= 1998 &&
  ram >= 256 && ram <= 3998;

  if (!isValid) {
    showModal();
    return;
  }

  

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      battery_power: battery_power,
      px_height: px_height,
      px_width: px_width,
      ram: ram
    })
  })
  .then(response => response.json())
  .then(data => {
    // Show tier in resultBox
    const resultBox = document.getElementById("resultBox");
    const resultText = document.getElementById("result");
    const revealBtn = document.getElementById("revealBtn");

    resultText.innerHTML = `<strong>Predicted Price Range:</strong> ${data.predicted_price_range}`;
    // Show upgrade suggestion if available
    const upgradeTipBox = document.getElementById("upgradeTipBox");
    const upgradeTip = document.getElementById("upgradeTip");

    if (data.upgrade_tip) {
      // Keep line breaks
      upgradeTip.innerHTML = `<span style="white-space: pre-line;">💡 ${data.upgrade_tip}</span>`;
      upgradeTipBox.style.display = "block";
    
      latestSuggestion = null; // No single upgrade to apply anymore
    } else {
      upgradeTipBox.style.display = "none";
      latestSuggestion = null;
    }
    
    resultBox.style.display = "block";
    revealBtn.style.display = "inline-block";

    // Store full response
    latestData = data;

    // Reset market cards if they were shown before
    document.getElementById("marketSection").classList.add("hidden");
    document.querySelectorAll('.market-card').forEach(card => {
      card.innerHTML = "";
      card.classList.remove("flip-in");
    });
  })
  .catch(error => {
    console.error("Prediction error:", error);
    document.getElementById("result").innerText = "Something went wrong.";
  });
}

function revealMarket() {
    if (!latestData || !latestData.market_examples) return;
  
    const examples = latestData.market_examples;
  
    for (let i = 0; i < 3; i++) {
      const cardData = examples[i];
      const cardEl = document.getElementById(`card${i + 1}`);
  
      if (cardData) {
        cardEl.innerHTML = `
          <h4>${cardData.name}</h4>
          <p>Battery: ${cardData.battery}</p>
          <p>RAM: ${cardData.ram}</p>
          <div class="phone-image-wrapper">
            <img src="images/${cardData.img}" alt="${cardData.name}" />
          </div>

          <a href="${cardData.link}" target="_blank" class="spec-btn">
            <span>🔍</span> View Specs
          </a>
        `;
      }
  
      cardEl.classList.add("flip-in");
    }
  
    document.getElementById("marketSection").classList.remove("hidden");}
  
  const inputs = document.querySelectorAll("#battery_power, #px_height, #px_width, #ram");
  const predictBtn = document.getElementById("predictBtn");

  function checkInputs() {
    const allFilled = Array.from(inputs).every(input => input.value.trim() !== "");
    predictBtn.disabled = !allFilled;
    predictBtn.style.opacity = allFilled ? 1 : 0.6;
    predictBtn.style.cursor = allFilled ? "pointer" : "not-allowed";
  
    const tooltip = document.getElementById("tooltipText");
    tooltip.style.display = allFilled ? "none" : "inline-block";
  }
  
  
  // Listen for input changes
  inputs.forEach(input => {
    input.addEventListener("input", checkInputs);
  });
  
  // Initial check on page load
  checkInputs();
  