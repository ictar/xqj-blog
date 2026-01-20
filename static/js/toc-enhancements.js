document.addEventListener("DOMContentLoaded", function () {
  /* ----------------------------------------------------
     1. Reading Progress Bar
     ---------------------------------------------------- */
  // Create progress bar element dynamically
  var progressBar = document.createElement("div");
  progressBar.id = "reading-progress";
  document.body.appendChild(progressBar);

  // Styling for progress bar is in toc.css (or can be set here)

  /* ----------------------------------------------------
     2. TOC Scroll Spy
     ---------------------------------------------------- */

  // Select only the header links in the TOC
  var tocLinks = document.querySelectorAll(".sidebar-toc a");
  var headers = [];

  // Identify all headers in the main content that match TOC links
  if (tocLinks.length > 0) {
    tocLinks.forEach(function (link) {
      if (!link.getAttribute("href").startsWith("#")) return; // Skip non-hash links

      var targetId = link.getAttribute("href").substring(1);
      // Escape special characters (like :) for querySelector using CSS.escape if needed,
      // but simpler to use getElementById which is faster and robust
      var targetHeader = document.getElementById(targetId);

      if (targetHeader) {
        headers.push({
          link: link,
          header: targetHeader,
        });
      }
    });
  }

  // Throttling function to prevent scroll firing too often
  var ticking = false;

  window.addEventListener("scroll", function () {
    if (!ticking) {
      window.requestAnimationFrame(function () {
        updateProgress();
        updateTOC();
        ticking = false;
      });
      ticking = true;
    }
  });

  function updateProgress() {
    var winScroll =
      document.body.scrollTop || document.documentElement.scrollTop;
    var height =
      document.documentElement.scrollHeight -
      document.documentElement.clientHeight;
    var scrolled = (winScroll / height) * 100;
    progressBar.style.width = scrolled + "%";
  }

  function updateTOC() {
    if (headers.length === 0) return;

    var scrollPos = window.scrollY || document.documentElement.scrollTop;
    // Add a small offset (e.g. 100px) so the header activates a bit before reaching top
    var offset = 150;

    // Find the current active header
    // We iterate backwards; the first header we find that is "above" the offset line is the active one
    var activeIndex = -1;

    for (var i = 0; i < headers.length; i++) {
      // If current header is *above* the cut-off line
      if (headers[i].header.offsetTop <= scrollPos + offset) {
        activeIndex = i;
      } else {
        // Since headers are ordered, once we find one below, we can stop
        break;
      }
    }

    // Reset all
    headers.forEach(function (item) {
      item.link.classList.remove("active-toc-link");
      // Optional: remove active class from parent li if desired
      item.link.parentElement.classList.remove("active-toc-li");
    });

    // Set Active
    if (activeIndex !== -1) {
      var activeItem = headers[activeIndex];
      activeItem.link.classList.add("active-toc-link");
      activeItem.link.parentElement.classList.add("active-toc-li");

      // Auto-scroll the TOC container if needed to keep active link in view
      // Only if sidebar (desktop)
      var sidebar = document.querySelector(".sidebar-toc");
      if (sidebar && window.innerWidth >= 1200) {
        // Simple logic: keep active link roughly in middle if possible or at least connected
        // activeItem.link.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        // 'nearest' is usually less jumpy than 'center'
      }
    }
  }

  // Initialize once
  updateProgress();
  updateTOC();
});
