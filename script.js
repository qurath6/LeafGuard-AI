
const imageInput = document.getElementById("leafImage");
const preview = document.getElementById("preview");
const analyzeButton = document.getElementById("analyzeButton");
const result = document.getElementById("result");

imageInput.addEventListener("change", function () {
    const file = this.files[0];
    result.textContent = "";

    if (!file) {
        preview.style.display = "none";
        return;
    }

    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
});

analyzeButton.addEventListener("click", async function () {
    const file = imageInput.files[0];

    if (!file) {
        result.textContent = "Please upload a leaf image first.";
        return;
    }

    // Open a blank tab immediately to avoid popup blocking
    const reportTab = window.open("about:blank", "_blank");

    if (!reportTab) {
        result.textContent = "Please allow pop-ups for this website.";
        return;
    }

    const formData = new FormData();
    formData.append("leafImage", file);

    result.textContent = "Analyzing your leaf...";
    analyzeButton.disabled = true;

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            reportTab.close();
            result.textContent = data.error || "Analysis failed.";
            return;
        }

        const reader = new FileReader();

        reader.onload = function () {
            data.image = reader.result;

            // Share report data with the new tab
            localStorage.setItem("leafReport", JSON.stringify(data));

            reportTab.location.href = "/result";
            result.textContent = "Analysis completed! Your report is open in a new tab.";
        };

        reader.readAsDataURL(file);

    } catch (error) {
        reportTab.close();
        result.textContent = "Unable to connect to the Python backend.";
    } finally {
        analyzeButton.disabled = false;
    }
});

