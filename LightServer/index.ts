//const app = require("./express-server-config")
import { AddressInfo } from "node:net";
import app from "./express-server-config"
import db from "./db-config"
const port = process.env.PORT || 8081;

// Connect Data Base
db

// Start web server
let server = app.listen(port, () => {
  console.log(`️[⚡ server]: Server is running at https://localhost:${(server.address() as AddressInfo)?.port}`);
});


