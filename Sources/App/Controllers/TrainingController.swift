import Vapor
import RQTFoundation
import PythonKit

struct TrainingController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let train = routes.grouped("train")
        train.get(use: trainEmbedding)
    }
    
    func trainEmbedding(req: Request) throws -> EventLoopFuture<String> {
        let req1 = RequirementVersionImpl(id: UUID.init().description, text: "B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        let req2 = RequirementVersionImpl(id: UUID.init().description, text: "B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        let req3 = RequirementVersionImpl(id: UUID.init().description, text: "B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times. B.J. Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        return try [req1, req2, req3].computeEmbedding(req: req, pathToExistingLM: nil, pathToTrainedLM: "trained/", max_epochs: 1).flatMap { (_) -> EventLoopFuture<String> in
            return try! trainModel(req: req)
        }
    }
    
    func trainModel(req: Request) throws -> EventLoopFuture<String> {
        let req1 = RequirementVersionImpl(id: UUID.init().description, text: "BJ Penn was a great fighter. Unfortunately, he got hit too many times.", source: UUID.init().description, createdAt: Date())
        let req2 = RequirementVersionImpl(id: UUID.init().description, text: "George Washington went to Washington. Sam Houston stayed home.", source: UUID.init().description, createdAt: Date())
        let req3 = RequirementVersionImpl(id: UUID.init().description, text: "Ja Morant wanted John Wall to do the Dougie in return to DC.", source: UUID.init().description, createdAt: Date())
        let req4 = RequirementVersionImpl(id: UUID.init().description, text: "Self-proclaimed crypto skeptic Max Levchin says Affirm may have to consider cryptocurrencies if Bitcoin's popularity continues to grow", source: UUID.init().description, createdAt: Date())
        let tag1 = RequirementTagImpl(id: nil, target: req1.id!, span: (0, 7), attribute: "NER", value: "Person", createdAt: nil)
        let tag2 = RequirementTagImpl(id: nil, target: req2.id!, span: (0, 17), attribute: "NER", value: "Person", createdAt: nil)
        let tag3 = RequirementTagImpl(id: nil, target: req2.id!, span: (25, 35), attribute: "GEO", value: "City", createdAt: nil)
        return try [req1, req2, req3, req4].computeAllModels(req: req, tags: [tag1, tag2, tag3], pathsToLMs: ["trained/"], testSplit: 0.25, devSplit: 0.25, epochs: 10).map { (_) -> (String) in
            print(Trainer.infer(for: [req1, req2, req3, req4]))
            return "training complete"
        }
    }
}
