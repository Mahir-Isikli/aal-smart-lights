import LampGroupModel from "../models/LampGroup/LampGroup.model";
import LampController from "./Lamp.controller";
import LampGroup from "../models/LampGroup/LampGroup.interface";
import LampConfig from "../models/Lampconfig/LampConfig.interface";

export default class GroupController {

    private lampController: LampController
    constructor(lampController: LampController) {
        this.lampController = lampController
    }

    /**
     * Generate fitting light configs for a group based on what lamps are currently available
     * @param fromGroupId
     */
    async generateLightConfigs(fromGroupId: string) {
        // fetch group from db
        const group: (LampGroup | null) = await LampGroupModel.findOne({ id: fromGroupId })
        let lamps = []
        let configs: LampConfig[] = []

        // check if group exists
        if (!group) throw ("Error, no group found with id " + fromGroupId)

        // find lamps that fulfill our criteria (correct location and category)
        lamps.push(... await this.lampController.fetchLampsByFunction(group.location, group.category))

        // generate configs based on color and intensity
        configs = lamps.map(lamp => Object({
            id: "",
            lampId: lamp.id,
            turnedOn: true,
            color: group.color,
            intensity: group.intensity
        }))

        return configs
    }
}
