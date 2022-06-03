//const app = require("./express-server-config")
import { AddressInfo } from "node:net";
import app from "./express-server-config"
const port = process.env.PORT || 8081;

// Start web server
let server = app.listen(port, () => {
  console.log("Routes: ", app.routes)
  console.log(`⚡️[server]: Server is running at https://localhost:${(server.address() as AddressInfo)?.port}`);
});
