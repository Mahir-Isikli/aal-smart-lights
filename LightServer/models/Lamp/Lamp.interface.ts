import {GroupCategory, GroupLocation} from "../LampGroup/LampGroup.interface";

export default interface Lamp {
    id: string,
    type: string,
    ip: string,
    hasColor: boolean,
    hasIntensity: boolean,
    category: GroupCategory,
    location: GroupLocation
}

