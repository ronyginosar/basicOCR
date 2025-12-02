const express = require('express');
const path = require('path');
const app = express();
const port = process.env.PORT || 8000; // Use environment variable or default to 3000

// Serve static files from a 'public' directory
app.use(express.static(path.join(__dirname)));

// Define a route to serve the index.html file for the root URL
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
  console.log(`Access the website at http://localhost:${port}`);
});