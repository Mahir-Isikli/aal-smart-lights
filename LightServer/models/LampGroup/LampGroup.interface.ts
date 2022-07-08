export default interface LampGroup {
    id: string,
    name: string,
    location: GroupLocation,
    category: GroupCategory,
    color: string,
    intensity: number
}

export enum GroupLocation {
    top,
    bottom,
    table,
    ceiling,
    chair,
    floor,
    rear
}

// FOR DOCU: 
// https://courses.cs.washington.edu/courses/cse458/05au/reading/3point_lighting.pdf
export enum GroupCategory {
    KeyLight, // main brightes light in the scene 
    FillLight, // softens and extends illumination provided by the scene 
    BackLight // creates a defining edge to seperate the subject from the background, Ambient Light
}

