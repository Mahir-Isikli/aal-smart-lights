"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const dotenv_1 = __importDefault(require("dotenv"));
const axios_1 = __importDefault(require("axios"));
dotenv_1.default.config();
const app = (0, express_1.default)();
const cors = require('cors');
const port = 8080; //process.env.PORT;
const BACKEND_IP = "192.168.10.193:80";
app.use(cors({ credentials: true, origin: true }));
app.use(express_1.default.static("public"));
app.get('/', (req, res) => {
    res.send('Express + TypeScript Server');
});
app.get('/lights/toggle/', (req, res) => {
    console.log("Toggling lights!");
    const status = req.query.status == "on";
    console.log("Toggeling on: ", status);
    axios_1.default.put("http://" + BACKEND_IP + "/api/8C2FF47893/lights/14:b4:57:ff:fe:72:35:7d-01/state", { on: status })
        .then(() => {
        res.status(200);
        res.send("Ok");
    })
        .catch((err) => {
        console.error("Error in toggle!" + err);
        res.status(500);
        res.send("Error! :(");
    });
});
app.listen(port, () => {
    console.log(`⚡️[server]: Server is running at https://localhost:${port}`);
});
// /lights/toggle 
