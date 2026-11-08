(() => {
    "use strict";

    function initializeFaqCards() {
        document.querySelectorAll(".faq-card").forEach((card) => {
            // Avoid adding duplicate handlers if this script is initialized twice.
            if (card.dataset.faqReady === "true") return;

            const front = card.querySelector(".faq-front");
            const back = card.querySelector(".faq-back");
            if (!front || !back) return;

            card.dataset.faqReady = "true";

            const setExpanded = (expanded) => {
                card.classList.toggle("is-flipped", expanded);
                card.setAttribute("aria-expanded", String(expanded));
                front.setAttribute("aria-hidden", String(expanded));
                back.setAttribute("aria-hidden", String(!expanded));
            };

            setExpanded(card.getAttribute("aria-expanded") === "true");

            card.addEventListener("click", () => {
                setExpanded(card.getAttribute("aria-expanded") !== "true");
            });

            card.addEventListener("keydown", (event) => {
                if (event.key === "Escape" && card.getAttribute("aria-expanded") === "true") {
                    event.preventDefault();
                    setExpanded(false);
                }
            });
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initializeFaqCards, { once: true });
    } else {
        initializeFaqCards();
    }
})();
