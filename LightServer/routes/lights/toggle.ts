/**
 * @openapi
 * /lights/toggle:
 *   get:
 *     summary: Toggle lights
 *     parameters:
 *       - in: query
 *         name: status
 *         schema:
 *           enum: ["on", "off"]
 *     responses:
 *       200:
 *         description: Ok
 *       400:
 *         description: Bad Request
 */

import express, { Request, Response } from 'express'
import axios from 'axios'

const BACKEND_IP = process.env.BACKEND_IP || "localhost:80" //"192.168.10.193:80" 

const router = express.Router()

router.get('/toggle', (req: Request, res: Response) => {
  const status = req.query.status == "on";
  console.log("Toggling lights! (" + (status ? "on" : "off") + ")");


  axios.put("http://" + BACKEND_IP + "/api/8C2FF47893/lights/14:b4:57:ff:fe:72:35:7d-01/state", { on: status })
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

export default router
