const express = require("express");
const path = require("path");
const bcrypt = require("bcrypt");

const app = express();

// Use EJS as the view engine
app.set("view engine", 'ejs');
//static file

app.use('/public',express.static("public"));
app.use('/views',express.static("views"));
app.use('/src',express.static("src"));

app.get("/consumer", (req, res) => {
    res.render("consumer");
});

app.get("/crop_trace", (req, res) => {
    res.render("crop_trace");
});


app.get("/farmer_page", (req, res) => {
    res.render("farmer_page");
});                                                                             

const port = 5000;
app.listen(port, () => {
    console.log(`Server running on Port: ${port}`);
});