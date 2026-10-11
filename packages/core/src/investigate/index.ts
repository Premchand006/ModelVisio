// AI performance investigator: deterministic evidence (pure TS).
export { buildEvidence, fixesResolving, type EvidenceOptions } from "./evidence";
export {
  layerRoofline, prepareLayers, rooflineFromPrepared,
  type LayerTime, type LayerRooflineResult, type PreparedLayers, type RooflineOpts,
} from "./layerRoofline";
