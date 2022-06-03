import SceneController from "./Scene.controller";

export default class EventController {
    // Verwaltet Liste von Events
    //  - kann hinzufügen, löschen, modifizieren
    private events: Array<LightEvent>
    private scenesController: SceneController

    constructor(scenesController: SceneController) {
        const sitDown: LightEvent = {
            id: "sitDown",
            scenes: ["luke1", "strip1"]
        }

        this.events = [sitDown]
        this.scenesController = scenesController
    }

    /**
     *  löst Event mit EventID aus
     */
    handleEvent(eventID: String): boolean {
        console.log("Handling Event: ", eventID)
        const event = this.events.find(event => event.id === eventID)
        if (!event || !eventID) return false

        return this.scenesController.triggerScene(eventID)
    }
}
