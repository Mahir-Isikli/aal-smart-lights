import SceneController from "./Scene.controller";
import LightEvent from "../models/LightEvent/LightEvent.model"

export default class EventController {
    // Verwaltet Liste von Events
    //  - kann hinzufügen, löschen, modifizieren
    //private events: Array<LightEvent>
    private scenesController: SceneController

    constructor(scenesController: SceneController) {
        /*
        const sitDown: LightEvent = {
            id: "sitDown",
            scenes: ["bright_lights", "strip1"]
        }

        this.events = [sitDown]

         */

        this.scenesController = scenesController
    }

    /**
     *  löst Event mit EventID aus
     */
    handleEvent(eventID: String): Promise<undefined> {
        return new Promise<undefined>((resolve, reject) => {
            console.log("[Event Controller] Handling Event: ", eventID)
            //LightEvent.find().then(events => console.log("Events: " + events))
            //const event = this.events.find(event => event.id === eventID)

            LightEvent.findOne({id: eventID}).then(event => {
                if (event === null) {
                    reject("No event found with ID: " + eventID)
                    return
                }

                Promise.allSettled(event.scenes.map(scene => this.scenesController.triggerScene(scene)))
                    .then(results => {
                        if (results.some(result => result.status === "rejected")) reject()
                        else resolve(undefined)
                    })
            }).catch(() => reject("No event found with ID: " + eventID))




        })
    }
}
