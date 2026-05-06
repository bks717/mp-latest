const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public')); // This will host your map.html file!

// Setup Multer to handle incoming image uploads in memory
const upload = multer({ storage: multer.memoryStorage() });

// The Bridge Endpoint: Catches image from frontend -> Sends to Python AI -> Returns Map Data
app.post('/api/analyze-satellite', upload.single('satelliteImage'), async (req, res) => {
    try {
        console.log("🛰️ Node Server received image from frontend. Forwarding to AI Engine...");

        // 1. Package the image into a form to send to Python
        const formData = new FormData();
        formData.append('file', req.file.buffer, { filename: req.file.originalname });

        // 2. Fire it at your local FastAPI server (which should be running on port 8000)
        const aiResponse = await axios.post('http://127.0.0.1:8000/predict', formData, {
            headers: { ...formData.getHeaders() }
        });

        console.log("✅ AI Engine replied with coordinates! Sending back to frontend.");
        
        // 3. Send the GeoJSON back to the web browser
        res.json(aiResponse.data);

    } catch (error) {
        console.error("🚨 Error communicating with AI Engine:", error.message);
        res.status(500).json({ error: "Failed to process image through AI." });
    }
});

app.listen(PORT, () => {
    console.log(`🌍 Main Dashboard Server running on http://localhost:${PORT}`);
});