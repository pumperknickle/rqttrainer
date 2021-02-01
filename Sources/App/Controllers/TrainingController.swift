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
        return try [req1, req2, req3].computeEmbedding(req: req, pathToExistingLM: nil, pathToTrainedLM: "trained/").map { (_) -> (String) in
            return "trained/"
        }
    }
}
