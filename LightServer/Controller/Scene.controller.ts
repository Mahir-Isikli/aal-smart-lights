import LampController from "./Lamp.controller";

export default class SceneController {
    // kennt alle scenes
    // kann scenes erstellen, löschen, modifizieren
    private scenes: Array<LightScene>
    private lampController: LampController
    constructor(lampController: LampController) {
        const luke1: LightScene = {
            id: "sitDown",
            lampConfigs: ["lukeOn"]
        }
        const strip1: LightScene = {
            id: "strip1",
            lampConfigs: ["stripOn"]
        }

        this.scenes = [luke1, strip1]
        this.lampController = lampController
    }

    /**
     * löst Scene mit SceneID aus, übergibt alle Lampconfigs an LampController
     * @param sceneID
     */
    triggerScene(sceneID: String): boolean {
        console.log("Triggering scene: ", sceneID)
        const scene = this.scenes.find(s => s.id === sceneID)
        let success = true

        if (!sceneID) return false

        for (const config in scene?.lampConfigs) {
            console.log("will exec: ", config)
            success = success && this.lampController.executeLampConfig(config)
        }

        return success
    }
}
