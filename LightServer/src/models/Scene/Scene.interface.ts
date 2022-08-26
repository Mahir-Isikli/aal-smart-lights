/**
 * A Scene is a set of Groups and Configs that will be executed when the scene gets triggered
 */
export default interface LightScene {
    id: String,
    lampConfigs: Array<String>,
    lampGroups: Array<String>
}
