impl<'a> Resolver<'a> {
    fn declaration(&self, ty: SemanticTypeIdV1) -> ResultV1<&'a SemanticTypeDeclV1> {
        self.owner
            .source_semantic()
            .types()
            .get(ty.index() as usize)
            .ok_or(ReferenceErrorV1::Unavailable("source type coordinate"))
    }
    fn is_u32(&self, ty: SemanticTypeIdV1) -> bool {
        self.declaration(ty).is_ok_and(|decl| {
            decl.rust_type_kind() == SemanticRustTypeKindV1::Ordinary
                && matches!(
                    decl.shape(),
                    SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                        signed: false,
                        bits: 32
                    })
                )
        })
    }
    fn is_f32(&self, ty: SemanticTypeIdV1) -> bool {
        self.declaration(ty).is_ok_and(|decl| {
            decl.rust_type_kind() == SemanticRustTypeKindV1::Ordinary
                && matches!(
                    decl.shape(),
                    SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float { bits: 32 })
                )
        })
    }
    fn zst(&self, ty: SemanticTypeIdV1) -> bool {
        self.declaration(ty)
            .is_ok_and(|decl| decl.layout().size_bytes() == Some(0))
    }

    fn exact_lane_field(&self, ty: SemanticTypeIdV1, field: u32) -> ResultV1<bool> {
        let SemanticTypeShapeV1::Aggregate(fields) = self.declaration(ty)?.shape() else {
            return refuse("current lane carrier is not its original aggregate");
        };
        if fields.fields().len() > 8 {
            return refuse("current lane field census");
        }
        let mut lane = None;
        for (index, ty) in fields.fields().iter().copied().enumerate() {
            if self.zst(ty) {
                continue;
            }
            if !self.is_u32(ty) {
                return refuse("current lane has non-u32 runtime state");
            }
            unique(&mut lane, index as u32)?;
        }
        Ok(lane == Some(field))
    }
    // Address-taken arithmetic snapshots are not promoted SSA. Authenticate
    // their actual unique source initializer; do not invent an SSA producer.
    // A consuming call reads the reference holder too. Ending storage after
    // Borrow alone is insufficient: every original later referent/holder read
    // must be disconnected from this kill, including paths around backedges.
}
