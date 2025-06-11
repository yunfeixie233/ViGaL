// Function to load HTML components
async function loadComponent(elementId, componentPath) {
    try {
      const response = await fetch(componentPath);
      const html = await response.text();
      document.getElementById(elementId).innerHTML = html;
    } catch (error) {
      console.error(`Error loading component ${componentPath}:`, error);
    }
  }
  
  // Load all components when DOM is ready
  document.addEventListener('DOMContentLoaded', async function() {
    const components = [
      { id: 'sidebar', path: './components/sidebar.html' },
      { id: 'header', path: './components/header.html' },
      { id: 'teaser', path: './components/teaser.html' },
      { id: 'snake-game', path: './components/snake-game.html' },
      { id: 'game-pipeline', path: './components/game-pipeline.html' },
      { id: 'evaluation-benchmark', path: './components/evaluation-benchmark.html' },
      { id: 'in-distribution-results', path: './components/in-distribution-results.html' },
      { id: 'out-of-domain-results', path: './components/out-of-domain-results.html' },
      { id: 'different-games-benefit', path: './components/different-games-benefit.html' },
      { id: 'games-visualization', path: './components/games-visualization.html' },
      { id: 'case-study', path: './components/case-study.html' },
      { id: 'footer', path: './components/footer.html' }
    ];
  
    // Load all components
    for (const component of components) {
      await loadComponent(component.id, component.path);
    }
  });