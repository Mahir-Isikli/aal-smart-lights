import SceneController from "./Scene.controller";
import LightEvent from "../models/LightEvent/LightEvent.model"

/**
 * Manages "Events": A Event is a relation between one EventID and many Scenes.
 *
 * @param sceneController: Reference to the SceneController instance
 */
export default class EventController {
    private scenesController: SceneController

    constructor(scenesController: SceneController) {
        this.scenesController = scenesController
    }

    /**
     *  Trigger event with the specified ID by looking it up in the DB
     *
     *  @param eventID String that
     */
    handleEvent(eventID: String): Promise<undefined> {
        return new Promise<undefined>((resolve, reject) => {
            console.log("[Event Controller] Handling Event: ", eventID)

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
