import express, { Express, Request, Response } from 'express';
import dotenv from 'dotenv';
import axios from 'axios'; 


dotenv.config();

const app: Express = express();
const cors = require('cors') 
const port = process.env.PORT || 8080; //;
const BACKEND_IP = process.env.BACKEND_IP || "localhost:80" //"192.168.10.193:80" 

app.use(cors({credentials: true, origin: true}))
app.use(express.static("public")); 
 
app.get('/', (req: Request, res: Response) => {
  res.send('Express + TypeScript Server');
});

app.get('/lights/toggle/', (req: Request, res: Response) => {
  const status = req.query.status == "on"; 
  console.log("Toggling lights! (" + (status ? "on" : "off") + ")");
  
  
  axios.put("http://"+ BACKEND_IP +"/api/8C2FF47893/lights/14:b4:57:ff:fe:72:35:7d-01/state", {on: status})
    .then(() => {
      res.status(200)
      res.send("Ok")
    })
    .catch((err) => {
      console.error("Error in toggle!" + err)
      res.status(500)
      res.send("Error! :(")
     })
}); 

app.listen(port, () => {
  console.log(`⚡️[server]: Server is running at https://localhost:${port}`);
});
