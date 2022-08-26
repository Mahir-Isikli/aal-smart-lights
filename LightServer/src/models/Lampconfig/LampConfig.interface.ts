/**
 * A Config defined how a lamp should be adjusted
 */
export default interface LampConfig {
    id: string,
    lampId: string,
    turnedOn: boolean,
    color: string,
    intensity: number
}
