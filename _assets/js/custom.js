document.addEventListener('DOMContentLoaded', function() {
  const codeBlocks = document.querySelectorAll('.franklin-content pre>code');

  codeBlocks.forEach(codeBlock => {
    // Create a button element
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'Copy';

    // Append the button to the parent of codeBlock (which is <pre>)
    const pre = codeBlock.parentElement;
    pre.classList.add('code-block'); // Add a class for styling
    pre.style.position = 'relative'; // Ensure the parent is positioned relatively
    pre.insertBefore(button, codeBlock);

    // Add click event listener to the button
    button.addEventListener('click', () => {
      // Copy the text content of the code block
      const textToCopy = codeBlock.textContent;
      navigator.clipboard.writeText(textToCopy).then(() => {
        // Provide feedback to the user
        button.textContent = 'Copied!';
        setTimeout(() => {
          button.textContent = 'Copy';
          button.style.backgroundColor = '#017da5'; // Reset background color after text reset
        }, 2000);
      }).catch(err => {
        console.error('Failed to copy text: ', err);
      });
    });
  });
});


// document.addEventListener('DOMContentLoaded', (event) => {
//     const codeBlocks = document.querySelectorAll('pre>code');
//
//     codeBlocks.forEach((codeBlock) => {
//         const button = document.createElement('button');
//         button.innerText = 'Copy';
//         button.style.borderImage = 'linear-gradient(white, grey) 1';
//         button.style.borderRadius = '50%';  // Oval border radius
//         button.className = 'copy-button';
//
//         button.addEventListener('click', () => {
//             const code = codeBlock.innerText;
//             navigator.clipboard.writeText(code).then(() => {
//                 button.innerText = 'Copied!';
//                 button.style.backgroundColor = ''; // Reset background color
//                 setTimeout(() => {
//                     button.innerText = 'Copy';
//                     button.style.backgroundColor = ''; // Reset background color after text reset
//                 }, 2000);
//             });
//         });
//
//         codeBlock.parentElement.insertBefore(button, codeBlock);
//     });
// });
