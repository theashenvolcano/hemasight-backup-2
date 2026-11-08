
const SETTINGS_KEY = "anemiaAccessibilitySettings";

const DEFAULT_SETTINGS = {
    theme: "light",
    colorTheme: "burgundy-cream",
    scheme: "default",
    font: "arial",
    textSize: "medium"
};

const COLOR_THEMES = ["burgundy-cream", "sage-ivory", "medical-blue", "teal-mint", "lavender-white", "high-contrast"];
const AVAILABLE_FONTS = ["arial", "verdana", "georgia", "opendyslexic", "inter", "roboto", "nunito-sans", "lato", "atkinson-hyperlegible", "times-new-roman"];

function normalizeSettings(saved) {
    const settings = { ...DEFAULT_SETTINGS, ...(saved && typeof saved === "object" ? saved : {}) };
    if (COLOR_THEMES.includes(saved?.colorTheme)) {
        settings.colorTheme = saved.colorTheme;
    } else if (saved?.scheme === "soft") {
        settings.colorTheme = "sage-ivory";
    } else if (saved?.scheme === "high-contrast") {
        settings.colorTheme = "high-contrast";
    } else if (saved?.scheme === "default" || !saved?.scheme) {
        settings.colorTheme = "burgundy-cream";
    } else {
        settings.colorTheme = "burgundy-cream";
    }
    settings.scheme = "default";
    if (!["light", "dark"].includes(settings.theme)) settings.theme = "light";
    if (!AVAILABLE_FONTS.includes(settings.font)) settings.font = "arial";
    if (!["small", "medium", "large", "extra-large"].includes(settings.textSize)) settings.textSize = "medium";
    return settings;
}

function getSettings() {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        const parsed = saved ? JSON.parse(saved) : {};
        const settings = normalizeSettings(parsed);
        if (saved && !COLOR_THEMES.includes(parsed?.colorTheme)) {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        }
        return settings;
    } catch (error) {
        return normalizeSettings({});
    }
}

function applySettings(settings) {
    const root = document.documentElement;

    root.dataset.theme = settings.theme;
    root.dataset.colorTheme = settings.colorTheme;
    root.dataset.font = settings.font;
    root.dataset.textSize = settings.textSize;
}

function saveSettings(settings) {
    try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (error) {
        console.warn("Could not save accessibility settings.", error);
    }

    applySettings(settings);
}

function initializeAccessibilityPage() {
    const fontSelect = document.getElementById("font-select");
    const textSizeSelect = document.getElementById("text-size-select");
    const colorThemeRadios = Array.from(
        document.querySelectorAll('input[name="colorTheme"]')
    );
    const themeSelect = document.getElementById("theme-select");
    const resetButton = document.getElementById("reset-settings");
    const resetCard = document.getElementById("reset-settings-card");
    const status = document.getElementById("settings-status");

    // Other pages may not have accessibility controls.
    if (!fontSelect || !textSizeSelect || !colorThemeRadios.length || !themeSelect) {
        return;
    }

    function updateControls(settings) {
        fontSelect.value = settings.font;
        textSizeSelect.value = settings.textSize;
        colorThemeRadios.forEach((radio) => {
            radio.checked = radio.value === settings.colorTheme;
        });
        themeSelect.value = settings.theme;
    }

    function readControls() {
        return {
            font: fontSelect.value,
            textSize: textSizeSelect.value,
            colorTheme: colorThemeRadios.find((radio) => radio.checked)?.value
                || DEFAULT_SETTINGS.colorTheme,
            scheme: "default",
            theme: themeSelect.value
        };
    }

    function updateResetVisibility(settings) {
        if (resetCard) {
            resetCard.hidden = settings.theme === DEFAULT_SETTINGS.theme
                && settings.colorTheme === DEFAULT_SETTINGS.colorTheme
                && settings.font === DEFAULT_SETTINGS.font
                && settings.textSize === DEFAULT_SETTINGS.textSize;
        }
    }

    function handleChange() {
        const settings = readControls();
        saveSettings(settings);
        updateResetVisibility(settings);

        if (status) {
            status.textContent =
                "Your accessibility settings have been applied and saved.";
        }
    }

    const currentSettings = getSettings();

    updateControls(currentSettings);
    applySettings(currentSettings);
    updateResetVisibility(currentSettings);

    fontSelect.addEventListener("change", handleChange);
    textSizeSelect.addEventListener("change", handleChange);
    colorThemeRadios.forEach((radio) => {
        radio.addEventListener("change", handleChange);
    });
    themeSelect.addEventListener("change", handleChange);

    if (resetButton) {
        resetButton.addEventListener("click", () => {
            const defaults = { ...DEFAULT_SETTINGS };

            updateControls(defaults);
            saveSettings(defaults);
            updateResetVisibility(defaults);

            if (status) {
                status.textContent = "Settings reset to their defaults.";
            }
        });
    }
}

// Apply saved settings on every page that loads this script.
applySettings(getSettings());

if (document.readyState === "loading") {
    document.addEventListener(
        "DOMContentLoaded",
        initializeAccessibilityPage
    );
} else {
    initializeAccessibilityPage();
}
