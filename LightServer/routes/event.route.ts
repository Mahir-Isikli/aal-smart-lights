/**
 * @openapi
 * /event
 *   get: Disptach Event with ID: eventID
 *     summary:
 *     parameters:
 *       - in: id
 *         name:
 *         schema:
 *           eventID: String
 *     responses:
 *       200:
 *         description: Ok
 *       400:
 *         description: invalid eventID
 */

import express, { Request, Response } from 'express'
import LampController from "../Controller/Lamp.controller"
import SceneController from "../Controller/Scene.controller"
import EventController from "../Controller/Event.controller"

//const BACKEND_IP = process.env.BACKEND_IP || "localhost:80" //"192.168.10.193:80"

const router = express.Router()

const lampController: LampController = new LampController()
const scenesController: SceneController = new SceneController(lampController)
const eventController: EventController = new EventController(scenesController)

router.get('/trigger', (req: Request, res: Response) => {
    const eventID = req.query.eventID || "";

    console.log("[Event API] Received event with ID: " + eventID)

    eventController.handleEvent(eventID as String)

    res.status(200)
    res.send("Ok")
});

export default router
