/**
 * LampGroups is a cluster of Lamps that can be grouped by their Location or Category. This eliminates the need to configure each lamp one by one.
 */
export default interface LampGroup {
    id: string,
    name: string,
    location: GroupLocation,
    category: GroupCategory,
    color: string,
    intensity: number
}

/**
 * Defnies where a Lamp is placed in the room
 */
export enum GroupLocation {
    top,
    bottom,
    table,
    ceiling,
    chair,
    floor,
    rear
}


/**
 * Defines what role the Lamp has in the Scene, refer to "Drei Punkte Beleuchtung" for more info
 */
export enum GroupCategory {
    KeyLight, // main brightes light in the scene 
    FillLight, // softens and extends illumination provided by the scene 
    BackLight // creates a defining edge to seperate the subject from the background, Ambient Light
}

