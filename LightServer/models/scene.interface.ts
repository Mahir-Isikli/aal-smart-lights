import { Schema, model } from 'mongoose';

interface LightScene {
    id: String,
    lampConfigs: Array<String>
}

const schema = new Schema<LightScene>({
    id: {type: String, required: true, unique: true},
    lampConfigs: Array
})

export default model("LightScene", schema)
