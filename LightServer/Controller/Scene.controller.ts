import LampController from "./Lamp.controller";
import LightSceneModel from "../models/Scene/Scene.model"
import GroupController from "./Group.controller";
import LampConfig from "../models/Lampconfig/LampConfig.interface";
import LampConfigModel from "../models/Lampconfig/LampConfig.model";
import LightScene from "../models/Scene/Scene.interface";

export default class SceneController {
    // kennt alle scenes und alle groups 
    // kann scenes und groups erstellen, löschen, modifizieren
 
    private lampController: LampController
    private groupController: GroupController
    constructor(lampController: LampController, groupController: GroupController) {
        this.lampController = lampController
        this.groupController = groupController
    }

    

    /**
     * löst Scene mit SceneID aus, übergibt alle Lampconfigs an LampController
     * @param sceneID
     */
    async triggerScene(sceneID: String) {
        console.log("[Scene Controller] Triggering scene: ", sceneID)

        // fetch scene from db
        const scene: (LightScene | null) = await LightSceneModel.findOne({id: sceneID})
        // make sure we have a valid Scene
        if (!scene)
            throw("Error: LightScene: " + sceneID + " not found")
        console.log("[Scene Controller] got scene: ", scene)

        // fetch configs from db
        const configs: LampConfig[] = []
        for (const configId of scene.lampConfigs) {
            const config: (LampConfig | null) = await LampConfigModel.findOne({id: configId})
            if (!config) continue
            configs.push(config)
        }

        // generate configs from group
        for (const groupId of scene.lampGroups) {
            const generatedConfigs: LampConfig[] = await this.groupController.generateLightConfigs(groupId as string)
            configs.push(... generatedConfigs)
        }

        // send configs to lamp controller for execution
        const promises: Array<Promise<void>> = configs.map(config => this.lampController.executeLampConfig(config))
        await Promise.allSettled(promises)
    }
}



