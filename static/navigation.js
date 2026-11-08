(() => {
    "use strict";

    document.querySelectorAll("[data-sliding-nav]").forEach((track) => {
        const links = Array.from(track.querySelectorAll("[data-nav-option]"));
        const slider = track.querySelector("[data-nav-slider]");
        const nav = track.closest(".shared-nav");
        const ariaCurrentIndex = links.findIndex((link) => link.getAttribute("aria-current") === "page");
        const renderedIndex = Number.parseInt(track.dataset.currentIndex, 10);
        const currentIndex = ariaCurrentIndex >= 0
            ? ariaCurrentIndex
            : (Number.isInteger(renderedIndex) ? renderedIndex : 0);
        const transitionKey = "anemiaNavigationFrom";
        let previousIndex = null;
        try {
            const savedIndex = Number.parseInt(sessionStorage.getItem(transitionKey), 10);
            sessionStorage.removeItem(transitionKey);
            if (Number.isInteger(savedIndex) && savedIndex >= 0 && savedIndex < links.length) {
                previousIndex = savedIndex;
            }
        } catch (_) {
            // Navigation still works if session storage is unavailable.
        }
        let pointer = null;
        let suppressClick = false;

        if (!slider || links.length < 2) return;

        const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
        const setPosition = (position) => {
            const bounded = clamp(position, 0, links.length - 1);
            track.style.setProperty("--slider-position", String(bounded));
            track.style.setProperty("--slider-offset", `${bounded * 100}%`);
        };
        const rememberPosition = (destination) => {
            if (destination === currentIndex) return;
            try {
                sessionStorage.setItem(transitionKey, String(currentIndex));
            } catch (_) {
                // The destination page will still render its active tab correctly.
            }
        };
        const rememberArrivalPosition = (position) => {
            try {
                sessionStorage.setItem(transitionKey, String(position));
            } catch (_) {
                // The destination page still renders its correct active position.
            }
        };
        const resetPosition = () => setPosition(currentIndex);

        if (previousIndex !== null && previousIndex !== currentIndex
            && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
            nav.classList.add("is-nav-initializing");
            setPosition(previousIndex);
            window.requestAnimationFrame(() => {
                nav.classList.remove("is-nav-initializing");
                window.requestAnimationFrame(resetPosition);
            });
        } else {
            resetPosition();
        }

        track.addEventListener("pointerdown", (event) => {
            if (event.button !== undefined && event.button !== 0) return;
            if (event.target.closest("a") && event.target.closest("a").classList.contains("settings-link")) return;
            if (currentIndex < 0) return;

            const sliderBounds = slider.getBoundingClientRect();

            pointer = {
                id: event.pointerId,
                startX: event.clientX,
                startY: event.clientY,
                grabOffsetX: event.clientX - sliderBounds.left,
                dragging: false,
                cancelled: false
            };
        });

        track.addEventListener("pointermove", (event) => {
            if (!pointer || event.pointerId !== pointer.id) return;
            const dx = event.clientX - pointer.startX;
            const dy = event.clientY - pointer.startY;

            if (!pointer.dragging && !pointer.cancelled) {
                if (Math.abs(dy) > 8 && Math.abs(dy) > Math.abs(dx)) {
                    pointer.cancelled = true;
                    return;
                }
                if (Math.abs(dx) < 8 || Math.abs(dx) <= Math.abs(dy)) return;

                pointer.dragging = true;
                nav.classList.add("is-nav-dragging");
                track.setPointerCapture(event.pointerId);
            }

            if (!pointer.dragging) return;
            event.preventDefault();
            const rect = track.getBoundingClientRect();
            const padding = 4;
            const slotWidth = (rect.width - padding * 2) / links.length;
            setPosition((event.clientX - rect.left - padding - pointer.grabOffsetX) / slotWidth);
        }, { passive: false });

        const finishPointer = (event, cancelled = false) => {
            if (!pointer || event.pointerId !== pointer.id) return;
            const wasDragging = pointer.dragging;
            if (wasDragging) {
                nav.classList.remove("is-nav-dragging");
                const destination = cancelled
                    ? currentIndex
                    : Math.round(Number(track.style.getPropertyValue("--slider-position")) || 0);
                suppressClick = true;
                window.setTimeout(() => { suppressClick = false; }, 400);
                if (destination >= 0 && destination < links.length && !cancelled) {
                    setPosition(destination);
                    if (destination !== currentIndex) {
                        // The pill has already reached its destination on this
                        // page, so don't replay the same movement after reload.
                        rememberArrivalPosition(destination);
                        const settleDelay = window.matchMedia("(prefers-reduced-motion: reduce)").matches
                            ? 0
                            : 380;
                        window.setTimeout(() => window.location.assign(links[destination].href), settleDelay);
                    }
                } else {
                    resetPosition();
                }
            }
            pointer = null;
        };

        track.addEventListener("pointerup", (event) => finishPointer(event));
        track.addEventListener("pointercancel", (event) => finishPointer(event, true));
        track.addEventListener("lostpointercapture", (event) => {
            // Touch pointers start captured by the pressed link. Changing capture
            // to the track bubbles a loss event from that link, which is a normal
            // handoff rather than a cancelled drag.
            if (event.target === track) finishPointer(event, true);
        });

        track.addEventListener("click", (event) => {
            if (!suppressClick) return;
            event.preventDefault();
            event.stopPropagation();
            suppressClick = false;
        }, true);

        links.forEach((link, index) => {
            link.addEventListener("click", () => rememberPosition(index));
            link.addEventListener("keydown", (event) => {
                let targetIndex = null;
                if (event.key === "ArrowRight") targetIndex = Math.min(links.length - 1, index + 1);
                if (event.key === "ArrowLeft") targetIndex = Math.max(0, index - 1);
                if (event.key === "Home") targetIndex = 0;
                if (event.key === "End") targetIndex = links.length - 1;
                if (targetIndex === null || targetIndex === index) return;
                event.preventDefault();
                rememberPosition(targetIndex);
                window.location.assign(links[targetIndex].href);
            });
        });
    });
})();
