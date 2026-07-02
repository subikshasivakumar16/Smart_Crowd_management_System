const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const bodyParser = require("body-parser");
const path = require("path");

const User = require("./models/User");

const app = express();

// Middleware
app.use(bodyParser.json());
app.use(express.urlencoded({ extended: true }));

// Static files
app.use(express.static(path.join(__dirname, "public")));


// ✅ FIXED MongoDB Connection (REMOVE OLD OPTIONS)
mongoose.connect("mongodb://127.0.0.1:27017/loginDB")
.then(() => console.log("✅ MongoDB Connected"))
.catch(err => console.log("❌ MongoDB Error:", err));


// Default route
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "login.html"));
});


// SIGNUP
app.post("/signup", async (req, res) => {
    const { email, password, role } = req.body;

    try {
        const existingUser = await User.findOne({ email });

        if (existingUser) {
            return res.json({ message: "User already exists" });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const newUser = new User({
            email,
            password: hashedPassword,
            role
        });

        await newUser.save();

        res.json({ message: "Signup successful" });

    } catch (err) {
        console.log(err);
        res.json({ message: "Error" });
    }
});


// LOGIN
app.post("/login", async (req, res) => {
    const { email, password } = req.body;

    try {
        const user = await User.findOne({ email });

        if (!user) {
            return res.json({ message: "User not found" });
        }

        const isMatch = await bcrypt.compare(password, user.password);

        if (!isMatch) {
            return res.json({ message: "Invalid password" });
        }

        res.json({
            message: "Login successful",
            role: user.role
        });

    } catch (err) {
        console.log(err);
        res.json({ message: "Error" });
    }
});


// Server
app.listen(5000, () => {
    console.log("🚀 Server running on http://localhost:5000");
});