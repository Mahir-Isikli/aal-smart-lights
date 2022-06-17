import LampController from "./Lamp.controller";
import LightScene from "../models/Scene/Scene.model"

export default class SceneController {
    // kennt alle scenes
    // kann scenes erstellen, löschen, modifizieren
    //private scenes: Array<LightScene>
    private lampController: LampController
    constructor(lampController: LampController) {
        /*
        const bright_lights: LightScene = {
            id: "bright_lights",
            lampConfigs: ["lukeOn"]
        }
        const strip1: LightScene = {
            id: "strip1",
            lampConfigs: ["stripOn"]
        }

        this.scenes = [bright_lights, strip1]

         */

        this.lampController = lampController
    }

    /**
     * löst Scene mit SceneID aus, übergibt alle Lampconfigs an LampController
     * @param sceneID
     */
    triggerScene(sceneID: String): Promise<undefined> {
        console.log("[Scene Controller] Triggering scene: ", sceneID)

        return new Promise<undefined>((resolve, reject) => {
            LightScene.findOne({id: sceneID}).then(scene => {
                // make sure we have a valid Scene
                if (!sceneID || !scene)
                    return reject("Error: LightScene: " + sceneID + " not found")

                console.log("[Scene Controller] got scene: ", scene)
                // execute all Configs of the Scene
                const promises: Array<Promise<undefined>> = scene.lampConfigs
                    .map(config => this.lampController.executeLampConfig(config))

                // throw error if an execution failed
                Promise.allSettled(promises).then(results => {
                    if (results.some(result => result.status === "rejected"))
                        reject("Error: Triggering scene " + sceneID)
                    else resolve(undefined)
                }).catch(() => reject("Error when executing Scene: " + sceneID))
            }).catch(() => reject("Error: LightScene: " + sceneID + " not found"))
        })
    }
}



