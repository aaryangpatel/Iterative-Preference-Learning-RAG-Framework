/**
 * Project site interactions for Preference-Learning RAG.
 *
 * Bootstraps smooth scrolling, section scrollspy, reveal-on-scroll,
 * the six-step preference loop stepper, and the benchmark system panel.
 */
const LOOP_STEPS = [
  {
    kicker: "01 · Retrieve",
    copy: "Pull supporting documents for the topic. The system queries a search API, so the full corpus does not need to live on disk.",
  },
  {
    kicker: "02 · Dual reports",
    copy: "Write two cited drafts from the same question bank: one abstractive, one extractive.",
  },
  {
    kicker: "03 · Judge",
    copy: "Compare the two drafts and keep the stronger one as champion.",
  },
  {
    kicker: "04 · Contrast",
    copy: "Write new questions for information the winner covered more completely than the loser.",
  },
  {
    kicker: "05 · Merge",
    copy: "Add those questions to the bank so later drafts cannot skip the same gaps.",
  },
  {
    kicker: "06 · Challenger",
    copy: "Write one improved draft and compare it with the champion. Repeat until the loop stops.",
  },
];

const SYSTEMS = [
  {
    kicker: "Proposed method",
    title: "preference_loop_full",
    copy: "Proposed method. Full loop: two opening drafts, contrastive questions, then challenger rounds until stop.",
  },
  {
    kicker: "Ablation",
    title: "preference_loop_1round",
    copy: "Ablation. Dual drafts and contrastive questions only. No later challenger rounds.",
  },
  {
    kicker: "Ablation",
    title: "crucible_dual_best",
    copy: "Ablation. Keeps the better of the first two drafts and stops.",
  },
  {
    kicker: "Baseline",
    title: "crucible_single",
    copy: "Baseline. One CRUCIBLE pass from the original question bank. No pairwise judge.",
  },
  {
    kicker: "Baseline",
    title: "vanilla_rag",
    copy: "Baseline. Questions come from the query alone, then a single extraction pass.",
  },
];

/**
 * Bind in-page anchors so nav and hero buttons scroll to their targets.
 */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const id = link.getAttribute("href");
      const target = id ? document.querySelector(id) : null;
      if (!target) {
        return;
      }
      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

/**
 * Mark the nav link for the section nearest the top of the viewport.
 */
function initScrollSpy() {
  const links = Array.from(document.querySelectorAll(".site-nav a"));
  const sections = links
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  const update = () => {
    const marker = window.scrollY + 120;
    let current = sections[0];
    sections.forEach((section) => {
      if (section.offsetTop <= marker) {
        current = section;
      }
    });
    links.forEach((link) => {
      link.classList.toggle("is-active", link.getAttribute("href") === `#${current.id}`);
    });
  };

  update();
  window.addEventListener("scroll", update, { passive: true });
}

/**
 * Fade in blocks tagged with `.reveal` when they enter the viewport.
 */
function initReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
  );

  document.querySelectorAll(".reveal").forEach((node) => observer.observe(node));
}

/**
 * Drive the six-step preference loop. `currentStep` stays in 0–5.
 *
 * Click a node or use ArrowLeft / ArrowRight when the loop panel is focused.
 */
function initLoopStepper() {
  const root = document.querySelector("[data-loop]");
  if (!root) {
    return;
  }

  const nodes = Array.from(root.querySelectorAll("[data-step]"));
  const kicker = root.querySelector("[data-loop-kicker]");
  const copy = root.querySelector("[data-loop-copy]");
  let currentStep = 0;

  const render = () => {
    const step = LOOP_STEPS[currentStep];
    nodes.forEach((node, index) => {
      node.classList.toggle("is-current", index === currentStep);
    });
    kicker.textContent = step.kicker;
    copy.textContent = step.copy;
  };

  nodes.forEach((node) => {
    node.addEventListener("click", () => {
      currentStep = Number(node.dataset.step);
      render();
    });
  });

  root.setAttribute("tabindex", "0");
  root.addEventListener("keydown", (event) => {
    if (event.key === "ArrowRight") {
      currentStep = (currentStep + 1) % LOOP_STEPS.length;
      render();
    }
    if (event.key === "ArrowLeft") {
      currentStep = (currentStep - 1 + LOOP_STEPS.length) % LOOP_STEPS.length;
      render();
    }
  });

  render();
}

/**
 * Update the benchmark detail panel when a system id is selected.
 * The first system is active on load.
 */
function initSystemFocus() {
  const root = document.querySelector("[data-systems]");
  if (!root) {
    return;
  }

  const items = Array.from(root.querySelectorAll("[data-system]"));
  const kicker = root.querySelector("[data-system-kicker]");
  const title = root.querySelector("[data-system-title]");
  const copy = root.querySelector("[data-system-copy]");
  let activeSystem = 0;

  const render = () => {
    const system = SYSTEMS[activeSystem];
    items.forEach((item, index) => {
      item.classList.toggle("is-active", index === activeSystem);
    });
    kicker.textContent = system.kicker;
    title.textContent = system.title;
    copy.textContent = system.copy;
  };

  items.forEach((item) => {
    item.addEventListener("click", () => {
      activeSystem = Number(item.dataset.system);
      render();
    });
  });

  render();
}

document.addEventListener("DOMContentLoaded", () => {
  initSmoothScroll();
  initScrollSpy();
  initReveal();
  initLoopStepper();
  initSystemFocus();
});
