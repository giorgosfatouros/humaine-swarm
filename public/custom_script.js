

// Placeholder for modifyLoginScreen function (not currently used)
function modifyLoginScreen() {
  // This function can be implemented if you want to customize the login screen
  console.log('Login screen modification placeholder');
}

function modifyChainlitWatermark() {
  let customWatermark = null;

  function updateWatermarkColor() {
    const originalParagraph = document.querySelector('.watermark p');
    if (originalParagraph && customWatermark) {
      const computedStyle = window.getComputedStyle(originalParagraph);
      customWatermark.style.color = computedStyle.color;
    }
  }

  const observer = new MutationObserver(() => {
    const watermarkContainer = document.querySelector('.watermark');

    if (watermarkContainer) {
      const originalLink = watermarkContainer.querySelector('a');
      const existingParagraph = watermarkContainer.querySelector('p');
      const chainlitIcon = originalLink.querySelector('svg');

      // Check if the watermark has already been modified
      if (!watermarkContainer.querySelector('.custom-watermark')) {
        if (existingParagraph && chainlitIcon) {
          const computedStyle = window.getComputedStyle(existingParagraph);

          // Create main container
          customWatermark = document.createElement('div');
          customWatermark.classList.add('custom-watermark');
          customWatermark.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            font-family: ${computedStyle.fontFamily};
            font-size: ${computedStyle.fontSize};
            font-weight: ${computedStyle.fontWeight};
            color: ${computedStyle.color};
          `;

          // Create top row container
          const topRow = document.createElement('div');
          topRow.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 8px;
            gap: 4px;
          `;

          // Create middle row for company and terms
          const middleRow = document.createElement('div');
          middleRow.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 6px 0;
            gap: 16px;
            font-size: 0.9em;
          `;

          // Move the original Chainlit content
          originalLink.parentElement.insertBefore(customWatermark, originalLink);
          topRow.appendChild(originalLink);

          // Create company link
          const companyLink = document.createElement('a');
          companyLink.href = 'https://humaine-horizon.eu/';
          companyLink.textContent = 'HEU HumAIne';
          companyLink.target = '_blank';
          companyLink.rel = 'noopener noreferrer';
          companyLink.style.cssText = `
            text-decoration: none;
            color: inherit;
            opacity: 0.8;
            transition: opacity 0.2s;
          `;
          companyLink.addEventListener('mouseenter', () => {
            companyLink.style.opacity = '1';
          });
          companyLink.addEventListener('mouseleave', () => {
            companyLink.style.opacity = '0.8';
          });

          // Create terms link
          const termsLink = document.createElement('a');
          termsLink.href = 'https://www.humaine-horizon.eu/terms-and-conditions';
          termsLink.textContent = 'Terms';
          termsLink.target = '_blank';
          termsLink.rel = 'noopener noreferrer';
          termsLink.style.cssText = `
            text-decoration: none;
            color: inherit;
            opacity: 0.8;
            transition: opacity 0.2s;
          `;
          termsLink.addEventListener('mouseenter', () => {
            termsLink.style.opacity = '1';
          });
          termsLink.addEventListener('mouseleave', () => {
            termsLink.style.opacity = '0.8';
          });

       

          // Create contact link
          const contactLink = document.createElement('a');
          contactLink.href = 'mailto:info@innov-acts.com';
          contactLink.textContent = 'Contact';
          contactLink.style.cssText = `
            text-decoration: none;
            color: inherit;
            opacity: 0.8;
            transition: opacity 0.2s;
          `;
          contactLink.addEventListener('mouseenter', () => {
            contactLink.style.opacity = '1';
          });
          contactLink.addEventListener('mouseleave', () => {
            contactLink.style.opacity = '0.8';
          });

          // Add divider
          const divider1 = document.createElement('span');
          divider1.textContent = '•';
          divider1.style.opacity = '0.5';

          const divider2 = document.createElement('span');
          divider2.textContent = '•';
          divider2.style.opacity = '0.5';

          const divider3 = document.createElement('span');
          divider3.textContent = '•';
          divider3.style.opacity = '0.5';

          // Populate middle row
          middleRow.appendChild(companyLink);
          middleRow.appendChild(divider1);
          middleRow.appendChild(termsLink);
          middleRow.appendChild(divider3);
          middleRow.appendChild(contactLink);

          // Create disclaimer text
          const disclaimer = document.createElement('div');
          disclaimer.style.cssText = `
            font-size: 0.8em;
            max-width: 800px;
            line-height: 1.4;
            opacity: 0.7;
            margin-top: 4px;
          `;
          disclaimer.textContent = 'Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.';

          // Update all links to use the computed color
          const links = [companyLink, termsLink, msLink, contactLink];
          links.forEach(link => {
            link.style.color = 'inherit';
          });

          // Update disclaimer to use the computed color
          disclaimer.style.color = 'inherit';
          disclaimer.style.opacity = '0.7';

          // Add all rows to the container
          customWatermark.appendChild(topRow);
          customWatermark.appendChild(middleRow);
          customWatermark.appendChild(disclaimer);

          // Replace the original watermark content with the new container
          watermarkContainer.innerHTML = '';
          watermarkContainer.appendChild(customWatermark);

          // Set up a MutationObserver to watch for color changes
          const colorObserver = new MutationObserver(updateWatermarkColor);
          colorObserver.observe(existingParagraph, {
            attributes: true,
            attributeFilter: ['style']
          });
        }
      }
    }
  });

  // Observe the body for changes, and don't disconnect the observer
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// Execute both functions when the DOM is ready
function executeModifications() {
  modifyLoginScreen();
  modifyChainlitWatermark();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', executeModifications);
} else {
  executeModifications();
}

// Additional check after a longer delay
setTimeout(() => {
  // Check if the user is logged in before attempting to modify the login screen
  if (!isUserLoggedIn()) {
    const rootContainer = document.querySelector('#root');

    if (rootContainer) {
      const loginContainer = rootContainer.querySelector('.MuiStack-root');

      if (loginContainer) {
        const logoImg = loginContainer.querySelector('img[alt="logo"]');
        const loginText = loginContainer.querySelector('.MuiTypography-root');

        if (!logoImg || !loginText) {
          modifyLoginScreen();
        }
      } else {
        modifyLoginScreen();
      }
    }
  }
}, 15000); // 15 seconds delay

// Define the isUserLoggedIn function outside to make it accessible globally
function isUserLoggedIn() {
  // Check for the element with id "new-chat-button" which appears when the user is logged in
  return !!document.querySelector('#new-chat-button');
}
