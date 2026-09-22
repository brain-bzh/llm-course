window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex|md-nav__link|md-ellipsis"
  }
};

document$.subscribe(() => {
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    if (MathJax.startup && MathJax.startup.output && MathJax.startup.output.clearCache) {
      MathJax.startup.output.clearCache();
    }
    if (MathJax.typesetClear) {
      MathJax.typesetClear();
    }
    if (MathJax.texReset) {
      MathJax.texReset();
    }
    MathJax.typesetPromise();
  }
});
