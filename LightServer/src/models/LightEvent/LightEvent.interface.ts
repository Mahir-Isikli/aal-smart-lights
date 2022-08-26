/**
 * A Event holds a set of Scenes that it will execute when the Events occurs
 */
export default interface LightEvent {
    id: String,
    scenes: Array<String>
}

