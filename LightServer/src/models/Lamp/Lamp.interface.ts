import {GroupCategory, GroupLocation} from "../LampGroup/LampGroup.interface";

/**
 * Models a Lamp by defining the communication protocol, IP and its capabilities
 */
export default interface Lamp {
    id: string,
    type: string,
    ip: string,
    hasColor: boolean,
    hasIntensity: boolean,
    category: GroupCategory,
    location: GroupLocation
}

