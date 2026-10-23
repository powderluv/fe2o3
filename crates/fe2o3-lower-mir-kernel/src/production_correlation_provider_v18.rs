// Both translation routes use this query surface. Only the legacy provider
// owns graph maps; the source-owned provider borrows the already-paid inventory.
trait SemanticAccessQueriesV18 {
    fn access_site(
        &self,
        location: FunctionOperationLocation,
        ordinal: u32,
    ) -> Option<SemanticAccessSiteV1>;
}

impl SemanticAccessQueriesV18 for BTreeMap<(FunctionOperationLocation, u32), SemanticAccessSiteV1> {
    fn access_site(
        &self,
        location: FunctionOperationLocation,
        ordinal: u32,
    ) -> Option<SemanticAccessSiteV1> {
        self.get(&(location, ordinal)).copied()
    }
}

trait RankedCorrelationQueriesV18 {
    fn source_index_identity(&self) -> Option<usize> {
        None
    }
    fn source_location_row(&self, _: (u32, u32)) -> Option<usize> {
        None
    }
    fn source(&self, site: SemanticAccessSiteV1) -> Option<&IndexedRankedAccessSourceV1>;
    fn conservative_source(
        &self,
        block: u32,
        statement: Option<u32>,
    ) -> Option<&IndexedRankedAccessSourceV1>;
    fn site(&self, location: (u32, u32)) -> Option<SemanticAccessSiteV1>;
    fn view(&self, value: ProductionRankedValueIdV1) -> Option<RankedViewDefinitionV1>;
    fn expression(
        &self,
        value: ProductionRankedValueIdV1,
    ) -> Option<(
        &ProductionSemanticExpressionV2,
        ProductionNumericalContractV2,
    )>;
    fn sources(&self) -> RankedSourcesV18<'_>;
}

enum RankedSourcesV18<'a> {
    Legacy(
        std::collections::btree_map::Iter<'a, SemanticAccessSiteV1, IndexedRankedAccessSourceV1>,
    ),
    Source(std::slice::Iter<'a, (SemanticAccessSiteV1, IndexedRankedAccessSourceV1)>),
}

impl<'a> Iterator for RankedSourcesV18<'a> {
    type Item = (&'a SemanticAccessSiteV1, &'a IndexedRankedAccessSourceV1);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Legacy(rows) => rows.next(),
            Self::Source(rows) => rows.next().map(|(site, source)| (site, source)),
        }
    }
}

impl RankedCorrelationQueriesV18 for RankedCorrelationIndexV1<'_> {
    fn source(&self, site: SemanticAccessSiteV1) -> Option<&IndexedRankedAccessSourceV1> {
        self.sources_by_site.get(&site)
    }
    fn conservative_source(
        &self,
        block: u32,
        statement: Option<u32>,
    ) -> Option<&IndexedRankedAccessSourceV1> {
        self.conservative_sources_by_statement
            .get(&(block, statement))
    }
    fn site(&self, location: (u32, u32)) -> Option<SemanticAccessSiteV1> {
        self.sites_by_ranked_location.get(&location).copied()
    }
    fn view(&self, value: ProductionRankedValueIdV1) -> Option<RankedViewDefinitionV1> {
        self.view_definitions.get(&value).copied()
    }
    fn expression(
        &self,
        value: ProductionRankedValueIdV1,
    ) -> Option<(
        &ProductionSemanticExpressionV2,
        ProductionNumericalContractV2,
    )> {
        self.semantic_expressions.get(&value).copied()
    }
    fn sources(&self) -> RankedSourcesV18<'_> {
        RankedSourcesV18::Legacy(self.sources_by_site.iter())
    }
}

struct SourceRankedIndexDataV18<'recipe> {
    sources: Vec<(SemanticAccessSiteV1, IndexedRankedAccessSourceV1)>,
    locations: Vec<usize>,
    conservative: Vec<(u32, Option<u32>, usize)>,
    views: Vec<(ProductionRankedValueIdV1, RankedViewDefinitionV1)>,
    expressions: Vec<(
        ProductionRankedValueIdV1,
        (
            &'recipe ProductionSemanticExpressionV2,
            ProductionNumericalContractV2,
        ),
    )>,
}

#[derive(Clone, Copy)]
enum RankedIndexSourcesV18<'a> {
    Original(&'a [ProductionRankedAccessSourceV1]),
    Qualified(&'a [ProductionSourceRankedAccessV18]),
}

impl RankedIndexSourcesV18<'_> {
    fn len(self) -> usize {
        match self {
            Self::Original(rows) => rows.len(),
            Self::Qualified(rows) => rows.len(),
        }
    }

    fn source(self, index: usize) -> Option<ProductionRankedAccessSourceV1> {
        match self {
            Self::Original(rows) => rows.get(index).copied(),
            Self::Qualified(rows) => {
                let row = rows.get(index)?;
                let key = row.solver?;
                let mut source = row.projected;
                source.semantic_block = key.block;
                source.semantic_statement = key.statement;
                source.semantic_access_ordinal = key.ordinal;
                Some(source)
            }
        }
    }
}

fn source_ranked_site_key_v18(site: SemanticAccessSiteV1) -> [usize; 4] {
    // Option ordering must match the shared legacy BTree key ordering.
    [
        site.block as usize,
        usize::from(site.statement.is_some()),
        site.statement.unwrap_or(0) as usize,
        site.ordinal as usize,
    ]
}

impl<'recipe> SourceRankedIndexDataV18<'recipe> {
    fn build(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        recipe: &'recipe fe2o3_pliron::ProductionRankedKernelV1,
        sources: &[ProductionRankedAccessSourceV1],
        max_operations: usize,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        Self::build_sources(
            relation,
            recipe,
            RankedIndexSourcesV18::Original(sources),
            max_operations,
            budget,
        )
    }

    fn build_qualified(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        optimized: &ProductionOptimizedSourceCorrespondenceV18<'_>,
        root: usize,
        recipe: &'recipe fe2o3_pliron::ProductionRankedKernelV1,
        sources: &[ProductionSourceRankedAccessV18],
        max_operations: usize,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        relation.retain_query((|| {
            budget.reserve_storage(argument_sum_v1(&[
                size_of::<RankedIndexSourcesV18<'_>>(),
                size_of::<SourceOwnedResultV18<Self>>(),
            ])?)?;
            check_source_ranked_accesses_v18(relation, optimized, root, sources, budget)?;
            Self::build_sources(
                relation,
                recipe,
                RankedIndexSourcesV18::Qualified(sources),
                max_operations,
                budget,
            )
        })())
    }

    fn build_sources(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        recipe: &'recipe fe2o3_pliron::ProductionRankedKernelV1,
        sources: RankedIndexSourcesV18<'_>,
        max_operations: usize,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        relation.retain_query((|| {
            relation.query(budget)?;
            let mut views = 0usize;
            let mut expressions = 0usize;
            let mut operations = 0usize;
            for block in recipe.blocks() {
                budget.charge_work(1)?;
                for operation in block.operations() {
                    budget.charge_work(1)?;
                    operations = operations
                        .checked_add(1)
                        .ok_or(ArgumentResourceV1::Arithmetic)?;
                    if operations > max_operations {
                        return relation
                            .source
                            .missing("ranked metadata exceeds the original operation limit");
                    }
                    if matches!(
                        operation,
                        ProductionRankedOperationV1::View { .. }
                            | ProductionRankedOperationV1::ViewInSpace { .. }
                    ) {
                        views = views.checked_add(1).ok_or(ArgumentResourceV1::Arithmetic)?;
                    }
                    if matches!(
                        operation,
                        ProductionRankedOperationV1::SemanticExpression { .. }
                    ) {
                        expressions = expressions
                            .checked_add(1)
                            .ok_or(ArgumentResourceV1::Arithmetic)?;
                    }
                }
            }
            if sources.len() > DEFAULT_MAX_OPERATIONS_V1 {
                return relation
                    .source
                    .missing("ranked source metadata exceeds the original source limit");
            }
            budget.reserve_storage(argument_sum_v1(&[
                size_of::<Self>(),
                size_of::<CorrelationLedgerV18<'_, '_, '_>>(),
                size_of::<SourceCorrelationChargeV18<'_, '_, '_, '_>>(),
            ])?)?;
            let mut result = Self {
                sources: emission_vec_v1(sources.len(), budget)
                    .map_err(source_emission_error_v18)?,
                locations: emission_vec_v1(sources.len(), budget)
                    .map_err(source_emission_error_v18)?,
                conservative: emission_vec_v1(sources.len(), budget)
                    .map_err(source_emission_error_v18)?,
                views: emission_vec_v1(views, budget).map_err(source_emission_error_v18)?,
                expressions: emission_vec_v1(expressions, budget)
                    .map_err(source_emission_error_v18)?,
            };
            let limit = max_operations
                .checked_mul(UNSUPPORTED_INDEX_CORRELATION_STEPS_PER_OPERATION_V1)
                .ok_or(ArgumentResourceV1::Arithmetic)?;
            let ledger = CorrelationLedgerV18::new(budget, relation.source.cleanup);
            let mut charge = SourceCorrelationChargeV18 {
                ledger: &ledger,
                finite: UnsupportedIndexCorrelationBudgetV1 { remaining: limit },
                finite_denied: false,
            };
            let visited = visit_ranked_index_metadata_sources_v18(
                recipe,
                sources,
                max_operations,
                &mut charge,
                |event, _| {
                    match event {
                        RankedIndexEventV18::View(value, definition) => {
                            if result.views.len() >= views
                                || result.views.len() == result.views.capacity()
                            {
                                return None;
                            }
                            result.views.push((value, definition));
                        }
                        RankedIndexEventV18::Expression(value, expression, contract) => {
                            if result.expressions.len() >= expressions
                                || result.expressions.len() == result.expressions.capacity()
                            {
                                return None;
                            }
                            result.expressions.push((value, (expression, contract)));
                        }
                        RankedIndexEventV18::Access(source, indexed) => {
                            if result.sources.len() >= sources.len()
                                || result.sources.len() == result.sources.capacity()
                            {
                                return None;
                            }
                            result.sources.push((
                                SemanticAccessSiteV1 {
                                    block: source.semantic_block,
                                    statement: source.semantic_statement,
                                    ordinal: source.semantic_access_ordinal,
                                },
                                indexed,
                            ));
                        }
                    }
                    Some(())
                },
            );
            let resource = ledger.failure.get();
            let refused = ledger.inconsistent_inventory.get() || charge.finite_denied;
            drop(charge);
            drop(ledger);
            if let Some(error) = resource {
                return Err(error.into());
            }
            if visited.is_none()
                || refused
                || result.sources.len() != sources.len()
                || result.views.len() != views
                || result.expressions.len() != expressions
            {
                return relation
                    .source
                    .missing("ranked metadata has an incomplete shared operation/source decode");
            }
            private_array_heapsort_v1(
                &mut result.views,
                |row| [row.0.get() as usize],
                &mut SourceCorrespondenceWorkV18(budget),
                || ArgumentResourceV1::Arithmetic.into(),
            )?;
            private_array_heapsort_v1(
                &mut result.expressions,
                |row| [row.0.get() as usize],
                &mut SourceCorrespondenceWorkV18(budget),
                || ArgumentResourceV1::Arithmetic.into(),
            )?;
            private_array_heapsort_v1(
                &mut result.sources,
                |row| source_ranked_site_key_v18(row.0),
                &mut SourceCorrespondenceWorkV18(budget),
                || ArgumentResourceV1::Arithmetic.into(),
            )?;
            for duplicate in result
                .views
                .windows(2)
                .map(|pair| pair[0].0 == pair[1].0)
                .chain(
                    result
                        .expressions
                        .windows(2)
                        .map(|pair| pair[0].0 == pair[1].0),
                )
            {
                budget.charge_work(1)?;
                if duplicate {
                    return relation
                        .source
                        .missing("ranked metadata repeats a view or expression definition");
                }
            }
            let mut first = 0usize;
            while first < result.sources.len() {
                let start = result.sources[first].0;
                let mut end = first;
                let mut direct = None;
                let mut ambiguous = false;
                while end < result.sources.len() {
                    budget.charge_work(4)?;
                    let (site, source) = &result.sources[end];
                    if (site.block, site.statement) != (start.block, start.statement) {
                        break;
                    }
                    if usize::try_from(site.ordinal).ok() != end.checked_sub(first) {
                        return relation.source.missing(
                            "ranked source access ordinals repeat or omit an original occurrence",
                        );
                    }
                    if matches!(source.allocation, IndexedRankedAllocationV1::Direct(_)) {
                        if direct.replace(end).is_some() {
                            ambiguous = true;
                        }
                    }
                    result.locations.push(end);
                    end = end.checked_add(1).ok_or(ArgumentResourceV1::Arithmetic)?;
                }
                if let Some(row) = direct.filter(|_| !ambiguous) {
                    result
                        .conservative
                        .push((start.block, start.statement, row));
                }
                first = end;
            }
            private_array_heapsort_v1(
                &mut result.locations,
                |index| {
                    let row = result.sources[*index].1;
                    [row.ranked_block as usize, row.ranked_operation as usize]
                },
                &mut SourceCorrespondenceWorkV18(budget),
                || ArgumentResourceV1::Arithmetic.into(),
            )?;
            for pair in result.locations.windows(2) {
                budget.charge_work(4)?;
                let first = result.sources[pair[0]].1;
                let second = result.sources[pair[1]].1;
                if (first.ranked_block, first.ranked_operation)
                    == (second.ranked_block, second.ranked_operation)
                {
                    return relation
                        .source
                        .missing("ranked source metadata repeats a physical ranked operation");
                }
            }
            Ok(result)
        })())
    }
}

struct SourceRankedIndexV18<'a, 'recipe, 'b, 'w, 'c> {
    rows: &'a SourceRankedIndexDataV18<'recipe>,
    ledger: &'a CorrelationLedgerV18<'b, 'w, 'c>,
}

struct QualifiedSourceRankedDataV18<'relation, 'source, 'recipe, 'records> {
    shared: SourceRankedIndexDataV18<'recipe>,
    original: &'relation ProductionSourceCorrespondenceV18<'source>,
    optimized: &'relation ProductionOptimizedSourceCorrespondenceV18<'source>,
    root: usize,
    recipe: &'recipe fe2o3_pliron::ProductionRankedKernelV1,
    records: &'records [ProductionSourceRankedAccessV18],
    required: usize,
    slot: usize,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
}

impl<'relation, 'source, 'recipe, 'records>
    QualifiedSourceRankedDataV18<'relation, 'source, 'recipe, 'records>
{
    fn build(
        original: &'relation ProductionSourceCorrespondenceV18<'source>,
        optimized: &'relation ProductionOptimizedSourceCorrespondenceV18<'source>,
        root: usize,
        recipe: &'recipe fe2o3_pliron::ProductionRankedKernelV1,
        records: &'records [ProductionSourceRankedAccessV18],
        max_operations: usize,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        original.retain_query((|| {
            optimized_source_endpoints_v18(original, optimized, budget)?;
            budget.reserve_storage(argument_sum_v1(&[
                size_of::<Self>(),
                size_of::<SourceOwnedResultV18<Self>>(),
                size_of::<Option<(SourceEffectSiteV18, SemanticAccessSiteV1)>>(),
                size_of::<RankedIndexSourcesV18<'_>>(),
            ])?)?;
            let shared = SourceRankedIndexDataV18::build_qualified(
                original,
                optimized,
                root,
                recipe,
                records,
                max_operations,
                budget,
            )?;
            Ok(Self {
                shared,
                original,
                optimized,
                root,
                recipe,
                records,
                required: budget.storage(),
                slot: std::ptr::from_ref(budget) as usize,
                ledger: budget.work_ledger_identity_v1(),
            })
        })())
    }

    fn observe_custody(&self, budget: &ArgumentBudgetV1<'_>) -> SourceOwnedResultV18<()> {
        self.original.observe_custody(budget)?;
        if self.required > budget.storage()
            || self.slot != std::ptr::from_ref(budget) as usize
            || self.ledger != budget.work_ledger_identity_v1()
        {
            self.original.source.cleanup.deny_refund();
            return Err(ArgumentResourceV1::Accounting.into());
        }
        Ok(())
    }

    fn check(&self, budget: &mut ArgumentBudgetV1<'_>) -> SourceOwnedResultV18<()> {
        self.original.retain_query(self.observe_custody(budget))?;
        optimized_source_endpoints_v18(self.original, self.optimized, budget)?;
        self.original.source.root(self.root, budget)?;
        budget.charge_work(1)?;
        if self.shared.sources.len() != self.records.len() {
            return self
                .original
                .source
                .missing("qualified ranked index lost its immutable record census");
        }
        Ok(())
    }

    fn check_recipe(
        &self,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<()> {
        self.original.retain_query((|| {
            self.check(budget)?;
            budget.charge_work(1)?;
            if !std::ptr::eq(self.recipe, recipe) {
                return self
                    .original
                    .source
                    .missing("qualified ranked consumer substituted its recipe owner");
            }
            Ok(())
        })())
    }

    fn query_index<'a, 'b, 'w, 'c>(
        &'a self,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        ledger: &'a CorrelationLedgerV18<'b, 'w, 'c>,
    ) -> SourceOwnedResultV18<
        QualifiedSourceRankedIndexV18<'a, 'relation, 'source, 'recipe, 'records, 'b, 'w, 'c>,
    > {
        let checked = ledger.with_budget(|budget| {
            Ok(self.original.retain_query((|| {
                // Observe the retained rows before any new credit can hide a lost
                // floor, including when this ledger was created after underpayment.
                self.observe_custody(budget)?;
                budget.reserve_storage(qualified_source_ranked_query_headers_v18()?)?;
                self.check_recipe(recipe, budget)
            })()))
        })?;
        match checked {
            Ok(()) => Ok(QualifiedSourceRankedIndexV18 {
                bound: self,
                shared: SourceRankedIndexV18 {
                    rows: &self.shared,
                    ledger,
                },
            }),
            Err(error) => {
                match &error {
                    ProductionSourceOwnedViewErrorV18::Resource(resource) => {
                        ledger.fail(*resource);
                    }
                    _ => ledger.inconsistent_inventory.set(true),
                }
                Err(error)
            }
        }
    }
}

fn qualified_source_ranked_query_headers_v18() -> Result<usize, ArgumentResourceV1> {
    type Query<'a> = QualifiedSourceRankedIndexV18<'a, 'a, 'a, 'a, 'a, 'a, 'a, 'a>;
    argument_sum_v1(&[
        size_of::<Query<'_>>(),
        size_of::<SourceOwnedResultV18<Query<'_>>>(),
        size_of::<SourceOwnedResultV18<()>>(),
        size_of::<Result<SourceOwnedResultV18<()>, ArgumentResourceV1>>(),
        size_of::<&QualifiedSourceRankedDataV18<'_, '_, '_, '_>>(),
        size_of::<&fe2o3_pliron::ProductionRankedKernelV1>(),
    ])
}

struct QualifiedSourceRankedIndexV18<'a, 'relation, 'source, 'recipe, 'records, 'b, 'w, 'c> {
    bound: &'a QualifiedSourceRankedDataV18<'relation, 'source, 'recipe, 'records>,
    shared: SourceRankedIndexV18<'a, 'recipe, 'b, 'w, 'c>,
}

impl QualifiedSourceRankedIndexV18<'_, '_, '_, '_, '_, '_, '_, '_> {
    fn ready(&self) -> bool {
        let checked = self.shared.ledger.with_budget(|budget| {
            Ok(self.bound.original.retain_query((|| {
                self.bound.check(budget)?;
                budget.charge_work(1)?;
                if !std::ptr::eq(self.shared.rows, &self.bound.shared) {
                    return self
                        .bound
                        .original
                        .source
                        .missing("qualified ranked query substituted its decoded index");
                }
                Ok(())
            })()))
        });
        match checked {
            Ok(Ok(())) => true,
            Ok(Err(ProductionSourceOwnedViewErrorV18::Resource(error))) => {
                self.shared.ledger.fail(error);
                false
            }
            Ok(Err(_)) => {
                self.shared.ledger.inconsistent_inventory.set(true);
                false
            }
            Err(_) => false,
        }
    }

    fn original_site(&self, site: SourceEffectSiteV18) -> Option<SemanticAccessSiteV1> {
        if !self.ready() {
            return None;
        }
        let index = self.shared.find(
            self.bound.records,
            |row| source_flow_site_key_v18(row.original()),
            source_flow_site_key_v18(site),
        )?;
        self.bound.records.get(index)?.solver
    }
}

impl RankedCorrelationQueriesV18 for QualifiedSourceRankedIndexV18<'_, '_, '_, '_, '_, '_, '_, '_> {
    fn source_index_identity(&self) -> Option<usize> {
        self.ready()
            .then(|| self.shared.source_index_identity())
            .flatten()
    }
    fn source_location_row(&self, location: (u32, u32)) -> Option<usize> {
        self.ready()
            .then(|| self.shared.source_location_row(location))
            .flatten()
    }
    fn source(&self, site: SemanticAccessSiteV1) -> Option<&IndexedRankedAccessSourceV1> {
        self.ready().then(|| self.shared.source(site)).flatten()
    }
    fn conservative_source(
        &self,
        block: u32,
        statement: Option<u32>,
    ) -> Option<&IndexedRankedAccessSourceV1> {
        self.ready()
            .then(|| self.shared.conservative_source(block, statement))
            .flatten()
    }
    fn site(&self, location: (u32, u32)) -> Option<SemanticAccessSiteV1> {
        self.ready().then(|| self.shared.site(location)).flatten()
    }
    fn view(&self, value: ProductionRankedValueIdV1) -> Option<RankedViewDefinitionV1> {
        self.ready().then(|| self.shared.view(value)).flatten()
    }
    fn expression(
        &self,
        value: ProductionRankedValueIdV1,
    ) -> Option<(
        &ProductionSemanticExpressionV2,
        ProductionNumericalContractV2,
    )> {
        self.ready()
            .then(|| self.shared.expression(value))
            .flatten()
    }
    fn sources(&self) -> RankedSourcesV18<'_> {
        if self.ready() {
            self.shared.sources()
        } else {
            RankedSourcesV18::Source([].iter())
        }
    }
}

impl SourceRankedIndexV18<'_, '_, '_, '_, '_> {
    fn find<T, const N: usize>(
        &self,
        rows: &[T],
        key: impl Fn(&T) -> [usize; N],
        target: [usize; N],
    ) -> Option<usize> {
        let mut first = 0usize;
        let mut end = rows.len();
        while first < end {
            self.ledger
                .with_budget(|budget| {
                    budget.charge_work(N.checked_add(1).ok_or(ArgumentResourceV1::Arithmetic)?)
                })
                .ok()?;
            let middle = first + (end - first) / 2;
            if key(rows.get(middle)?) < target {
                first = middle + 1;
            } else {
                end = middle;
            }
        }
        self.ledger
            .with_budget(|budget| {
                budget.charge_work(N.checked_add(1).ok_or(ArgumentResourceV1::Arithmetic)?)
            })
            .ok()?;
        rows.get(first)
            .filter(|row| key(row) == target)
            .map(|_| first)
    }
}

impl RankedCorrelationQueriesV18 for SourceRankedIndexV18<'_, '_, '_, '_, '_> {
    fn source_index_identity(&self) -> Option<usize> {
        Some(self.rows as *const SourceRankedIndexDataV18<'_> as usize)
    }
    fn source_location_row(&self, location: (u32, u32)) -> Option<usize> {
        let index = self.find(
            &self.rows.locations,
            |index| {
                let row = self.rows.sources[*index].1;
                [row.ranked_block as usize, row.ranked_operation as usize]
            },
            [location.0 as usize, location.1 as usize],
        )?;
        self.rows.locations.get(index).copied()
    }
    fn source(&self, site: SemanticAccessSiteV1) -> Option<&IndexedRankedAccessSourceV1> {
        let index = self.find(
            &self.rows.sources,
            |row| source_ranked_site_key_v18(row.0),
            source_ranked_site_key_v18(site),
        )?;
        self.rows.sources.get(index).map(|row| &row.1)
    }
    fn conservative_source(
        &self,
        block: u32,
        statement: Option<u32>,
    ) -> Option<&IndexedRankedAccessSourceV1> {
        let key = |block: u32, statement: Option<u32>| {
            [
                block as usize,
                usize::from(statement.is_some()),
                statement.unwrap_or(0) as usize,
            ]
        };
        let index = self.find(
            &self.rows.conservative,
            |row| key(row.0, row.1),
            key(block, statement),
        )?;
        self.rows
            .conservative
            .get(index)
            .and_then(|row| self.rows.sources.get(row.2))
            .map(|row| &row.1)
    }
    fn site(&self, location: (u32, u32)) -> Option<SemanticAccessSiteV1> {
        let index = self.find(
            &self.rows.locations,
            |index| {
                let row = self.rows.sources[*index].1;
                [row.ranked_block as usize, row.ranked_operation as usize]
            },
            [location.0 as usize, location.1 as usize],
        )?;
        self.rows
            .locations
            .get(index)
            .and_then(|index| self.rows.sources.get(*index))
            .map(|row| row.0)
    }
    fn view(&self, value: ProductionRankedValueIdV1) -> Option<RankedViewDefinitionV1> {
        let index = self.find(
            &self.rows.views,
            |row| [row.0.get() as usize],
            [value.get() as usize],
        )?;
        self.rows.views.get(index).map(|row| row.1)
    }
    fn expression(
        &self,
        value: ProductionRankedValueIdV1,
    ) -> Option<(
        &ProductionSemanticExpressionV2,
        ProductionNumericalContractV2,
    )> {
        let index = self.find(
            &self.rows.expressions,
            |row| [row.0.get() as usize],
            [value.get() as usize],
        )?;
        self.rows.expressions.get(index).map(|row| row.1)
    }
    fn sources(&self) -> RankedSourcesV18<'_> {
        RankedSourcesV18::Source(self.rows.sources.iter())
    }
}

trait KirCorrelationGraphV18 {
    fn source_inventory_identity(&self) -> Option<usize> {
        None
    }
    fn source_definition_index(&self, _: ValueId) -> Option<usize> {
        None
    }
    fn unique_origin(
        &self,
        value: ValueId,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<ValueId>;
    fn operation(&self, location: FunctionOperationLocation) -> Option<&Operation>;
    fn definition(&self, value: ValueId) -> Option<&Operation>;
    fn definition_location(&self, value: ValueId) -> Option<FunctionOperationLocation>;
    fn block_operations(&self, block: BlockId) -> Option<&[Operation]>;
    fn inline_scalar(&self) -> &Gfx942InlineScalarCorrespondenceV30<'_>;
    fn scalar_parameter(&self, function: &Function, value: ValueId) -> Option<Option<u32>> {
        match function
            .body
            .as_ref()?
            .parameters
            .iter()
            .position(|candidate| *candidate == value)
        {
            Some(parameter) => Some(Some(u32::try_from(parameter).ok()?)),
            None => Some(None),
        }
    }
    fn scalar_result(
        &self,
        _: &Function,
        operation: &Operation,
        value: ValueId,
    ) -> Option<ProductionSemanticScalarTypeV2> {
        operation
            .results
            .iter()
            .find(|result| result.id == value)
            .and_then(|result| kir_semantic_scalar_v1(&result.ty))
    }
    fn scalar_type(
        &self,
        function: &Function,
        value: ValueId,
    ) -> Option<ProductionSemanticScalarTypeV2> {
        let body = function.body.as_ref()?;
        if let Some(parameter) = body
            .parameters
            .iter()
            .position(|candidate| *candidate == value)
        {
            return kir_semantic_scalar_v1(function.signature.parameters.get(parameter)?);
        }
        if let Some(parameter) = body
            .blocks
            .iter()
            .flat_map(|block| &block.parameters)
            .find(|parameter| parameter.id == value)
        {
            return kir_semantic_scalar_v1(&parameter.ty);
        }
        self.definition(value)?
            .results
            .iter()
            .find(|result| result.id == value)
            .and_then(|result| kir_semantic_scalar_v1(&result.ty))
    }
    fn scalar_leaf(
        &self,
        _: &Function,
        _: ValueId,
    ) -> Option<Option<NormalizedScalarExpressionV1>> {
        Some(None)
    }
    fn scalar_argument(
        &self,
        _: &Function,
        argument: u32,
        scalar: ProductionSemanticScalarTypeV2,
    ) -> Option<NormalizedScalarExpressionV1> {
        Some(NormalizedScalarExpressionV1::Symbol {
            symbol: PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2.checked_add(argument)?,
            scalar,
        })
    }

    fn scalar_binary_overflow(
        &self,
        _: &Function,
        _: ValueId,
        overflow: ProductionOverflowContractV2,
    ) -> Option<ProductionOverflowContractV2> {
        Some(overflow)
    }

    // Outer None is failure; inner None means this is not a block argument.
    // Some(0) is a genuine block argument with no incoming edges, not an origin.
    fn visit_incoming(
        &self,
        value: ValueId,
        visit: &mut dyn FnMut(ValueId) -> Option<()>,
    ) -> Option<Option<usize>>;
}

#[derive(Clone, Copy)]
struct GeneratedRecipeValuesV18 {
    destination_local: SemanticLocalIdV1,
    input: ValueId,
    output: ValueId,
}

// These are borrowed source attachments, not a second operation/CFG index.
// A source sequence may cross relocated ranges; its locations are never
// reconstructed by adding a source ordinal to a physical block ordinal.
enum NeutralRecipeOperationsV18<'a> {
    Legacy {
        operations: &'a [Operation],
        block: BlockId,
        first: usize,
    },
    Source {
        rows: &'a [SourceAttachmentV18],
        inventory: &'a fe2o3_kernel_analysis::CanonicalKirInventoryV18<'a>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        meter: &'a dyn NeutralRecipeMeterV18,
    },
    Optimized {
        block: u32,
        entries: usize,
        meter: &'a dyn NeutralRecipeMeterV18,
    },
}

trait NeutralRecipeMeterV18 {
    fn charge_recipe_step(&self) -> Option<()>;

    fn optimized_recipe_operation(
        &self,
        _block: u32,
        _ordinal: usize,
    ) -> Option<(&Operation, FunctionOperationLocation)> {
        None
    }
}

impl NeutralRecipeOperationsV18<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Legacy { operations, .. } => operations.len(),
            Self::Source { rows, .. } => rows.len(),
            Self::Optimized { entries, .. } => *entries,
        }
    }

    fn get(&self, ordinal: usize) -> Option<(&Operation, FunctionOperationLocation)> {
        match self {
            Self::Legacy {
                operations,
                block,
                first,
            } => Some((
                operations.get(ordinal)?,
                FunctionOperationLocation::new(*block, first.checked_add(ordinal)?),
            )),
            Self::Source {
                rows,
                inventory,
                function,
                meter,
            } => {
                meter.charge_recipe_step()?;
                let TileAttachmentLocationV29::Origin(TileScalarSourceV29::Operation(point)) =
                    rows.get(ordinal)?.location
                else {
                    return None;
                };
                if usize::try_from(function.0).ok()? != point.function {
                    return None;
                }
                let parent = inventory.functions().get(point.function)?;
                let ordinal = parent.blocks.start.checked_add(point.block)?;
                if ordinal >= parent.blocks.end {
                    return None;
                }
                let block = inventory.blocks().get(ordinal)?;
                if block.coordinate.function != *function
                    || usize::try_from(block.coordinate.block).ok()? != point.block
                {
                    return None;
                }
                Some((
                    block.block.operations.get(point.operation)?,
                    FunctionOperationLocation::new(block.block.id, point.operation),
                ))
            }
            Self::Optimized { block, meter, .. } => {
                meter.optimized_recipe_operation(*block, ordinal)
            }
        }
    }
}

trait GeneratedRecipeSourceV18 {
    fn values(&self, block: u32) -> Option<GeneratedRecipeValuesV18>;
    fn operations(&self, block: u32) -> Option<NeutralRecipeOperationsV18<'_>>;
    fn producer_contains(&self, block: u32, location: FunctionOperationLocation) -> Option<bool>;
    fn operation_origin(
        &self,
        body: &FunctionBody,
        kir: &dyn KirCorrelationGraphV18,
        value: ValueId,
    ) -> Option<ValueId>;
    fn charge(&self, amount: usize) -> Result<(), ProductionMirPlironTranslationErrorV1>;
    fn reserve(&self, bytes: usize) -> Result<(), ProductionMirPlironTranslationErrorV1>;
}

struct GeneratedRecipeChargeV18<'a>(&'a dyn GeneratedRecipeSourceV18);

impl PrivateArrayChargeV1 for GeneratedRecipeChargeV18<'_> {
    type Error = ProductionMirPlironTranslationErrorV1;
    fn charge_private_array_work(&mut self, amount: usize) -> Result<(), Self::Error> {
        self.0.charge(amount)
    }
}

// Every source allocation is prepaid. Credits for this scratch and its output
// stay with the containing correlation scope until all corresponding values
// have dropped; an inner error never refunds an enclosing denied-cleanup scope.
fn generated_rows_v18<T>(
    source: &dyn GeneratedRecipeSourceV18,
    count: usize,
) -> Result<Vec<T>, ProductionMirPlironTranslationErrorV1> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<T>>()))
        .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    source.charge(count)?;
    source.reserve(bytes)?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(count)
        .map_err(|_| ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
    source.reserve(
        rows.capacity()
            .checked_sub(count)
            .and_then(|extra| extra.checked_mul(std::mem::size_of::<T>()))
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?,
    )?;
    Ok(rows)
}

fn generated_push_v18<T>(
    rows: &mut Vec<T>,
    row: T,
) -> Result<(), ProductionMirPlironTranslationErrorV1> {
    if rows.len() == rows.capacity() {
        return Err(ProductionMirPlironTranslationErrorV1::ResourceLimit);
    }
    rows.push(row);
    Ok(())
}

fn generated_sort_v18<T, const N: usize>(
    rows: &mut [T],
    key: impl Fn(&T) -> [usize; N],
    source: &dyn GeneratedRecipeSourceV18,
) -> Result<(), ProductionMirPlironTranslationErrorV1> {
    private_array_heapsort_v1(rows, key, &mut GeneratedRecipeChargeV18(source), || {
        ProductionMirPlironTranslationErrorV1::ResourceLimit
    })
}

fn generated_range_v18<T, const N: usize>(
    rows: &[T],
    key: impl Fn(&T) -> [usize; N],
    target: [usize; N],
    source: &dyn GeneratedRecipeSourceV18,
) -> Result<std::ops::Range<usize>, ProductionMirPlironTranslationErrorV1> {
    let start = private_array_partition_v1(
        rows,
        &key,
        target,
        false,
        &mut GeneratedRecipeChargeV18(source),
    )?;
    let end = private_array_partition_v1(
        rows,
        key,
        target,
        true,
        &mut GeneratedRecipeChargeV18(source),
    )?;
    Ok(start..end)
}

struct LegacyGeneratedRecipeSourceV18<'a> {
    correspondence: &'a SemanticKirCorrespondenceV1,
    body: Option<&'a FunctionBody>,
    owner: SemanticFunctionIdV1,
    function: SemanticFunctionIdV1,
}

impl GeneratedRecipeSourceV18 for LegacyGeneratedRecipeSourceV18<'_> {
    fn charge(&self, _amount: usize) -> Result<(), ProductionMirPlironTranslationErrorV1> {
        Ok(())
    }
    fn reserve(&self, _bytes: usize) -> Result<(), ProductionMirPlironTranslationErrorV1> {
        Ok(())
    }

    fn operation_origin(
        &self,
        body: &FunctionBody,
        kir: &dyn KirCorrelationGraphV18,
        value: ValueId,
    ) -> Option<ValueId> {
        unique_ssa_operation_origin_v1(body, kir, value)
    }

    fn values(&self, block: u32) -> Option<GeneratedRecipeValuesV18> {
        let mut rows = self
            .correspondence
            .generated_terminator_values
            .iter()
            .filter(|row| {
                row.correspondence_owner == self.owner
                    && row.semantic_function == self.function
                    && row.semantic_block.index() == block
            });
        let row = rows.next()?;
        if rows.next().is_some() {
            return None;
        }
        Some(GeneratedRecipeValuesV18 {
            destination_local: row.destination_local,
            input: row.input,
            output: row.output,
        })
    }

    fn operations(&self, block: u32) -> Option<NeutralRecipeOperationsV18<'_>> {
        let span = self
            .correspondence
            .terminator_operation_spans()
            .iter()
            .find(|span| {
                span.correspondence_owner() == self.owner
                    && span.semantic_function() == self.function
                    && span.semantic_block().index() == block
            })?;
        let block = self
            .body?
            .blocks
            .iter()
            .find(|block| block.id == span.kernel_ir_block())?;
        let first = usize::try_from(span.first_operation_ordinal()).ok()?;
        let count = usize::try_from(span.operation_count()).ok()?;
        Some(NeutralRecipeOperationsV18::Legacy {
            operations: block.operations.get(first..first.checked_add(count)?)?,
            block: block.id,
            first,
        })
    }

    fn producer_contains(&self, block: u32, location: FunctionOperationLocation) -> Option<bool> {
        Some(
            self.correspondence
                .terminator_operation_spans()
                .iter()
                .any(|span| {
                    span.semantic_function() == self.function
                        && span.semantic_block().index() == block
                        && operation_span_contains_v1(
                            span.kernel_ir_block(),
                            span.first_operation_ordinal(),
                            span.operation_count(),
                            location,
                        )
                }),
        )
    }
}

type TranslationEffectLocationV18 = (
    SemanticAccessSiteV1,
    FunctionOperationLocation,
    u32,
    (u32, u32),
);

enum TranslationContractCollectorV18<'a, 'w, T> {
    Legacy(&'a mut Vec<T>),
    Count {
        count: &'a mut usize,
        budget: &'a mut ArgumentBudgetV1<'w>,
        failure: &'a mut Option<ArgumentResourceV1>,
    },
    Fill {
        rows: &'a mut Vec<T>,
        limit: usize,
        budget: &'a mut ArgumentBudgetV1<'w>,
        failure: &'a mut Option<ArgumentResourceV1>,
    },
}

impl<T> TranslationContractCollectorV18<'_, '_, T> {
    fn charge(&mut self, amount: usize) -> Result<(), ProductionMirPlironTranslationErrorV1> {
        match self {
            Self::Legacy(_) => Ok(()),
            Self::Count {
                budget, failure, ..
            }
            | Self::Fill {
                budget, failure, ..
            } => {
                if failure.is_some() {
                    return Err(ProductionMirPlironTranslationErrorV1::ResourceLimit);
                }
                budget.charge_work(amount).map_err(|error| {
                    **failure = Some(error);
                    ProductionMirPlironTranslationErrorV1::ResourceLimit
                })
            }
        }
    }

    fn push(&mut self, value: T) -> Result<(), ProductionMirPlironTranslationErrorV1> {
        self.charge(1)?;
        match self {
            Self::Legacy(rows) => rows.push(value),
            Self::Count { count, failure, .. } => {
                **count = count.checked_add(1).ok_or_else(|| {
                    **failure = Some(ArgumentResourceV1::Arithmetic);
                    ProductionMirPlironTranslationErrorV1::ResourceLimit
                })?;
            }
            Self::Fill { rows, limit, .. } => {
                if rows.len() >= *limit || rows.len() == rows.capacity() {
                    return Err(ProductionMirPlironTranslationErrorV1::ResourceLimit);
                }
                rows.push(value);
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
enum SourceTranslationCoreErrorV18 {
    Source(ProductionSourceOwnedViewErrorV18),
    Translation(ProductionMirPlironTranslationErrorV1),
}

impl From<ArgumentResourceV1> for SourceTranslationCoreErrorV18 {
    fn from(error: ArgumentResourceV1) -> Self {
        Self::Source(error.into())
    }
}
impl From<ProductionSourceOwnedViewErrorV18> for SourceTranslationCoreErrorV18 {
    fn from(error: ProductionSourceOwnedViewErrorV18) -> Self {
        Self::Source(error)
    }
}

fn source_translation_contracts_v18<T: Copy + Ord>(
    mut visit: impl FnMut(
        &mut TranslationContractCollectorV18<'_, '_, T>,
    ) -> Result<(), ProductionMirPlironTranslationErrorV1>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<Vec<T>, SourceTranslationCoreErrorV18> {
    budget.reserve_storage(argument_sum_v1(&[
        size_of::<Vec<T>>(),
        size_of::<TranslationContractCollectorV18<'_, '_, T>>(),
        std::mem::size_of_val(&visit),
        size_of::<Option<ArgumentResourceV1>>(),
        size_of::<Result<(), ProductionMirPlironTranslationErrorV1>>(),
        size_of::<Result<Vec<T>, SourceTranslationCoreErrorV18>>(),
        size_of::<usize>(),
    ])?)?;
    let mut count = 0usize;
    let mut failure = None;
    let result = visit(&mut TranslationContractCollectorV18::Count {
        count: &mut count,
        budget,
        failure: &mut failure,
    });
    if let Some(error) = failure {
        return Err(error.into());
    }
    result.map_err(SourceTranslationCoreErrorV18::Translation)?;
    let mut rows = emission_vec_v1(count, budget).map_err(source_emission_error_v18)?;
    let result = visit(&mut TranslationContractCollectorV18::Fill {
        rows: &mut rows,
        limit: count,
        budget,
        failure: &mut failure,
    });
    if let Some(error) = failure {
        return Err(error.into());
    }
    result.map_err(SourceTranslationCoreErrorV18::Translation)?;
    if rows.len() != count {
        return Err(
            ProductionSourceOwnedViewErrorV18::Binding("translation contract count/fill").into(),
        );
    }
    // These fixed-size contracts retain the historical multiset comparison.
    // CFG effect order is checked independently through the source relation.
    assert_origin_sort_v1(&mut rows, budget, |lhs, rhs, budget| {
        budget.charge_work(size_of::<T>())?;
        Ok(lhs.cmp(rhs))
    })
    .map_err(|error| match error {
        SemanticKirAssertOriginErrorV1::Resource(error) => error.into(),
        _ => ProductionSourceOwnedViewErrorV18::Binding("translation contract sort"),
    })?;
    Ok(rows)
}

fn source_translation_contract_multisets_v18(
    relation: &ProductionSourceCorrespondenceV18<'_>,
    optimized: Option<&ProductionOptimizedSourceCorrespondenceV18<'_>>,
    root: usize,
    body: &FunctionBody,
    recipe: &fe2o3_pliron::ProductionRankedKernelV1,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(usize, usize), SourceTranslationCoreErrorV18> {
    relation.query(budget)?;
    let function = match optimized {
        Some(optimized) => {
            optimized_source_endpoints_v18(relation, optimized, budget)?;
            optimized_source_root_function_v18(relation, optimized, root, budget)?
        }
        None => {
            let ordinal = relation.source.root(root, budget)?.1;
            relation.inventory.functions().get(ordinal).ok_or(
                ProductionSourceOwnedViewErrorV18::Binding("translation contracts original root"),
            )?
        }
    };
    budget.charge_work(1)?;
    if function
        .function
        .body
        .as_ref()
        .is_none_or(|actual| !std::ptr::eq(actual, body))
    {
        return Err(ProductionSourceOwnedViewErrorV18::Binding(
            "translation contracts exact endpoint body",
        )
        .into());
    }
    let floor = budget.storage();
    let run = |budget: &mut ArgumentBudgetV1<'_>| {
        let canonical = source_translation_contracts_v18(
            |collector| visit_kir_synchronization_contracts_v18(body, collector),
            budget,
        )?;
        let ranked = source_translation_contracts_v18(
            |collector| visit_ranked_synchronization_contracts_v18(recipe, collector),
            budget,
        )?;
        if !source_translation_contracts_equal_v18(&canonical, &ranked, budget)? {
            return Err(SourceTranslationCoreErrorV18::Translation(
                ProductionMirPlironTranslationErrorV1::SynchronizationMismatch,
            ));
        }
        let canonical_tensors = source_translation_contracts_v18(
            |collector| visit_kir_tensor_contracts_v18(body, collector),
            budget,
        )?;
        let ranked_tensors = source_translation_contracts_v18(
            |collector| visit_ranked_tensor_contracts_v18(recipe, collector),
            budget,
        )?;
        if !source_translation_contracts_equal_v18(&canonical_tensors, &ranked_tensors, budget)? {
            return Err(SourceTranslationCoreErrorV18::Translation(
                ProductionMirPlironTranslationErrorV1::TensorContractMismatch,
            ));
        }
        Ok((canonical.len(), canonical_tensors.len()))
    };
    type Result = std::result::Result<(usize, usize), SourceTranslationCoreErrorV18>;
    budget.reserve_storage(argument_sum_v1(&[
        std::mem::size_of_val(&run),
        size_of::<Result>(),
        size_of::<std::thread::Result<Result>>(),
        size_of::<std::panic::AssertUnwindSafe<Result>>(),
        size_of::<(usize, usize)>(),
    ])?)?;
    let result = scoped_source_attempt_v29(relation.source.cleanup, budget, floor, run)?;
    // Only two counts escape. All four buffers and captured temporaries are
    // gone before returning this scope's credit on the same ledger.
    budget.release_storage(
        budget
            .storage()
            .checked_sub(floor)
            .ok_or(ArgumentResourceV1::Accounting)?,
    )?;
    relation.query(budget)?;
    if let Some(optimized) = optimized {
        optimized_source_endpoints_v18(relation, optimized, budget)?;
    }
    Ok(result)
}

fn source_translation_contracts_equal_v18<T: Eq>(
    lhs: &[T],
    rhs: &[T],
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<bool, ArgumentResourceV1> {
    budget.charge_work(1)?;
    if lhs.len() != rhs.len() {
        return Ok(false);
    }
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        budget.charge_work(size_of::<T>())?;
        if lhs != rhs {
            return Ok(false);
        }
    }
    Ok(true)
}

struct SourceAllocationScratchV18 {
    inventory: usize,
    definitions: std::ops::Range<usize>,
    marks: Vec<usize>,
    epoch: usize,
    pending: Vec<ValueId>,
    pending_limit: usize,
    origin: Option<u32>,
}

impl SourceAllocationScratchV18 {
    fn build(
        inventory: &fe2o3_kernel_analysis::CanonicalKirInventoryV18<'_>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        budget.charge_work(4)?;
        let row = inventory
            .functions()
            .get(function.0 as usize)
            .filter(|row| row.coordinate == function)
            .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                "allocation scratch function",
            ))?;
        let definitions = row
            .definitions
            .end
            .checked_sub(row.definitions.start)
            .ok_or(ArgumentResourceV1::Arithmetic)?;
        let uses = row
            .uses
            .end
            .checked_sub(row.uses.start)
            .ok_or(ArgumentResourceV1::Arithmetic)?;
        let bindings = row
            .edge_arguments
            .end
            .checked_sub(row.edge_arguments.start)
            .ok_or(ArgumentResourceV1::Arithmetic)?;
        // Each definition is expanded once. Duplicate incoming/use incidences
        // still occupy pending slots and retain their normal pop-work charge.
        let pending_limit = argument_sum_v1(&[uses, bindings, 1])?;
        budget.reserve_storage(argument_sum_v1(&[
            size_of::<Self>(),
            size_of::<SourceOwnedResultV18<Self>>(),
            size_of::<Vec<usize>>(),
            size_of::<std::ops::Range<usize>>(),
            argument_product_v1(3, size_of::<usize>())?,
        ])?)?;
        let mut marks = emission_vec_v1(definitions, budget).map_err(source_emission_error_v18)?;
        budget.charge_work(definitions)?;
        marks.resize(definitions, 0);
        Ok(Self {
            inventory: inventory as *const fe2o3_kernel_analysis::CanonicalKirInventoryV18<'_>
                as usize,
            definitions: row.definitions.clone(),
            marks,
            epoch: 0,
            pending: emission_vec_v1(pending_limit, budget).map_err(source_emission_error_v18)?,
            pending_limit,
            origin: None,
        })
    }

    fn begin(
        &mut self,
        value: ValueId,
        kir: &dyn KirCorrelationGraphV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<()> {
        budget.charge()?;
        if kir.source_inventory_identity()? != self.inventory {
            return None;
        }
        budget.charge_many(self.pending.len())?;
        self.pending.clear();
        self.origin = None;
        self.epoch = match self.epoch.checked_add(1) {
            Some(epoch) => epoch,
            None => {
                budget.charge_many(self.marks.len())?;
                self.marks.fill(0);
                1
            }
        };
        self.push(value, budget)
    }

    fn push(&mut self, value: ValueId, budget: &mut dyn CorrelationChargeV18) -> Option<()> {
        budget.charge()?;
        if self.pending.len() >= self.pending_limit || self.pending.len() == self.pending.capacity()
        {
            return None;
        }
        self.pending.push(value);
        Some(())
    }

    fn insert(
        &mut self,
        value: ValueId,
        kir: &dyn KirCorrelationGraphV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<bool> {
        budget.charge()?;
        if kir.source_inventory_identity()? != self.inventory {
            return None;
        }
        let index = kir.source_definition_index(value)?;
        if !self.definitions.contains(&index) {
            return None;
        }
        let mark = self
            .marks
            .get_mut(index.checked_sub(self.definitions.start)?)?;
        let first = *mark != self.epoch;
        *mark = self.epoch;
        Some(first)
    }
}

enum AllocationWalkScratchV18<'a> {
    Legacy {
        visited: &'a mut BTreeSet<ValueId>,
        pending: Vec<ValueId>,
        origins: BTreeSet<u32>,
    },
    Source(&'a mut SourceAllocationScratchV18),
}

impl AllocationWalkScratchV18<'_> {
    fn parameter(
        &self,
        function: &Function,
        kir: &dyn KirCorrelationGraphV18,
        value: ValueId,
    ) -> Option<Option<usize>> {
        match self {
            Self::Legacy { .. } => Some(
                function
                    .body
                    .as_ref()?
                    .parameters
                    .iter()
                    .position(|parameter| *parameter == value),
            ),
            Self::Source(_) => Some(
                kir.scalar_parameter(function, value)?
                    .map(|index| index as usize),
            ),
        }
    }
    fn pop(&mut self) -> Option<ValueId> {
        match self {
            Self::Legacy { pending, .. } => pending.pop(),
            Self::Source(scratch) => scratch.pending.pop(),
        }
    }
    fn push(&mut self, value: ValueId, budget: &mut dyn CorrelationChargeV18) -> Option<()> {
        match self {
            Self::Legacy { pending, .. } => {
                pending.push(value);
                Some(())
            }
            Self::Source(scratch) => scratch.push(value, budget),
        }
    }
    fn insert(
        &mut self,
        value: ValueId,
        kir: &dyn KirCorrelationGraphV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<bool> {
        match self {
            Self::Legacy { visited, .. } => Some(visited.insert(value)),
            Self::Source(scratch) => scratch.insert(value, kir, budget),
        }
    }
    fn origin(&mut self, parameter: u32) -> Option<()> {
        match self {
            Self::Legacy { origins, .. } => {
                origins.insert(parameter);
                (origins.len() <= 1).then_some(())
            }
            Self::Source(scratch) => match scratch.origin {
                Some(old) if old != parameter => None,
                _ => {
                    scratch.origin = Some(parameter);
                    Some(())
                }
            },
        }
    }
    fn result(&self) -> Option<u32> {
        match self {
            Self::Legacy { origins, .. } => {
                let mut origins = origins.iter();
                let origin = *origins.next()?;
                origins.next().is_none().then_some(origin)
            }
            Self::Source(scratch) => scratch.origin,
        }
    }
}

enum TranslationCoreScratchV18 {
    Legacy {
        used: BTreeSet<(u32, u32)>,
        effects: Vec<TranslationEffectLocationV18>,
    },
    Source {
        ranked: usize,
        used: Vec<u8>,
        effects: Vec<TranslationEffectLocationV18>,
        effect_limit: usize,
        allocation: SourceAllocationScratchV18,
        scalar: SourceScalarVisitingV18,
    },
}

impl TranslationCoreScratchV18 {
    fn legacy() -> Self {
        Self::Legacy {
            used: BTreeSet::new(),
            effects: Vec::new(),
        }
    }

    fn source(
        ranked: &SourceRankedIndexDataV18<'_>,
        inventory: &fe2o3_kernel_analysis::CanonicalKirInventoryV18<'_>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        consumers: usize,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        budget.reserve_storage(argument_sum_v1(&[
            size_of::<Self>(),
            size_of::<SourceOwnedResultV18<Self>>(),
            size_of::<Vec<u8>>(),
        ])?)?;
        let mut used =
            emission_vec_v1(ranked.sources.len(), budget).map_err(source_emission_error_v18)?;
        budget.charge_work(ranked.sources.len())?;
        used.resize(ranked.sources.len(), 0u8);
        budget.charge_work(MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 + 1)?;
        Ok(Self::Source {
            ranked: ranked as *const SourceRankedIndexDataV18<'_> as usize,
            used,
            effects: emission_vec_v1(consumers, budget).map_err(source_emission_error_v18)?,
            effect_limit: consumers,
            allocation: SourceAllocationScratchV18::build(inventory, function, budget)?,
            scalar: SourceScalarVisitingV18 {
                rows: [None; MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2 + 1],
                length: 0,
            },
        })
    }

    fn use_location(
        &mut self,
        location: (u32, u32),
        index: &dyn RankedCorrelationQueriesV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<bool> {
        match self {
            Self::Legacy { used, .. } => Some(used.insert(location)),
            Self::Source { ranked, used, .. } => {
                budget.charge()?;
                if index.source_index_identity()? != *ranked {
                    return None;
                }
                let mark = used.get_mut(index.source_location_row(location)?)?;
                let first = *mark == 0;
                *mark = 1;
                Some(first)
            }
        }
    }

    fn contains_location(
        &self,
        location: (u32, u32),
        index: &dyn RankedCorrelationQueriesV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<bool> {
        match self {
            Self::Legacy { used, .. } => Some(used.contains(&location)),
            Self::Source { ranked, used, .. } => {
                budget.charge()?;
                if index.source_index_identity()? != *ranked {
                    return None;
                }
                Some(*used.get(index.source_location_row(location)?)? != 0)
            }
        }
    }

    fn push_effect(
        &mut self,
        effect: TranslationEffectLocationV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<()> {
        match self {
            Self::Legacy { effects, .. } => effects.push(effect),
            Self::Source {
                effects,
                effect_limit,
                ..
            } => {
                budget.charge()?;
                if effects.len() >= *effect_limit || effects.len() == effects.capacity() {
                    return None;
                }
                effects.push(effect);
            }
        }
        Some(())
    }

    fn effects(&self) -> &[TranslationEffectLocationV18] {
        match self {
            Self::Legacy { effects, .. } | Self::Source { effects, .. } => effects,
        }
    }

    fn conservative_effects(
        &self,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<usize> {
        match self {
            Self::Legacy { .. } => Some(
                recipe
                    .blocks()
                    .iter()
                    .flat_map(|block| block.operations())
                    .filter(|operation| {
                        matches!(
                            operation,
                            ProductionRankedOperationV1::AllocationEffect { .. }
                        )
                    })
                    .count(),
            ),
            Self::Source { .. } => {
                let mut count = 0usize;
                for block in recipe.blocks() {
                    budget.charge()?;
                    for operation in block.operations() {
                        budget.charge()?;
                        if matches!(
                            operation,
                            ProductionRankedOperationV1::AllocationEffect { .. }
                        ) {
                            count = count.checked_add(1)?;
                        }
                    }
                }
                Some(count)
            }
        }
    }

    fn external_parameter(
        &mut self,
        function: &Function,
        kir: &dyn KirCorrelationGraphV18,
        value: ValueId,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<u32> {
        match self {
            Self::Legacy { .. } => {
                external_allocation_parameter_v1(function, kir, value, &mut BTreeSet::new(), budget)
            }
            Self::Source { allocation, .. } => {
                allocation.begin(value, kir, budget)?;
                external_allocation_parameter_core_v18(
                    function,
                    kir,
                    &mut AllocationWalkScratchV18::Source(allocation),
                    budget,
                )
            }
        }
    }

    fn normalize_actual(
        &mut self,
        function: &Function,
        kir: &dyn KirCorrelationGraphV18,
        sites: &dyn SemanticAccessQueriesV18,
        value: ValueId,
        budget: &mut dyn CorrelationChargeV18,
        helpers: &mut native_helper_value_expansion_v1::NativeValueExpansion<'_, '_>,
    ) -> Option<NormalizedScalarExpressionV1> {
        match self {
            Self::Legacy { .. } => normalize_kir_expression_v1(
                function,
                kir,
                sites,
                value,
                0,
                &mut BTreeSet::new(),
                budget,
                helpers,
            ),
            Self::Source { scalar, .. } => {
                if scalar.length != 0 {
                    return None;
                }
                normalize_kir_expression_with_visiting_v18(
                    function, kir, sites, value, 0, scalar, budget, helpers,
                )
            }
        }
    }
}

enum TranslationEffectDispositionV18 {
    Ranked {
        site: SemanticAccessSiteV1,
        consumer: KirMemoryConsumerV1,
    },
    // Only the legacy provider uses its historical compiler-owned exclusions.
    LegacyCompilerStorage,
    // The source provider must first append an exact source/backing/effect row
    // to its pending obligation census. This never grants Storage safety.
    PendingSourceStorage,
}

trait TranslationSourceRelationV18 {
    fn check_contract_multisets(
        &mut self,
        function: &Function,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        helpers: &mut native_helper_value_expansion_v1::NativeValueExpansion<'_, '_>,
    ) -> Result<(usize, usize), ProductionMirPlironTranslationErrorV1>;
    fn check_effect_control_flow(
        &mut self,
        body: &FunctionBody,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        locations: &[TranslationEffectLocationV18],
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<(), ProductionMirPlironTranslationErrorV1>;
    fn semantic_sha256(&self) -> &[u8; 32];
    fn diagnostic_subject(&self) -> (SemanticFunctionIdV1, SemanticFunctionIdV1);
    fn effect(
        &mut self,
        consumer: KirMemoryConsumerV1,
        kir: &dyn KirCorrelationGraphV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<TranslationEffectDispositionV18, ProductionMirPlironTranslationErrorV1>;
    fn source_parameter(
        &mut self,
        function: &Function,
        parameter: u32,
        location: FunctionOperationLocation,
    ) -> Result<u32, ProductionMirPlironTranslationErrorV1>;
    fn check_private_access(
        &mut self,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        source: &IndexedRankedAccessSourceV1,
        consumer: KirMemoryConsumerV1,
        site: SemanticAccessSiteV1,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<(), ProductionMirPlironTranslationErrorV1>;
    fn requires_ranked_consumption(
        &mut self,
        site: SemanticAccessSiteV1,
        source: &IndexedRankedAccessSourceV1,
        private: bool,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<bool, ProductionMirPlironTranslationErrorV1>;
    fn legacy_private_recipe_exclusion(&self) -> bool;
}

struct LegacyTranslationSourceV18<'a> {
    semantic: Option<&'a AdmittedInertSemanticMirV1>,
    correspondence: &'a SemanticKirCorrespondenceV1,
    owner: SemanticFunctionIdV1,
    function: SemanticFunctionIdV1,
    sites: &'a BTreeMap<(FunctionOperationLocation, u32), SemanticAccessSiteV1>,
    private_arrays: Option<PrivateArrayFinalRelationV1<'a>>,
}

impl TranslationSourceRelationV18 for LegacyTranslationSourceV18<'_> {
    fn check_contract_multisets(
        &mut self,
        function: &Function,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        helpers: &mut native_helper_value_expansion_v1::NativeValueExpansion<'_, '_>,
    ) -> Result<(usize, usize), ProductionMirPlironTranslationErrorV1> {
        let body = function
            .body
            .as_ref()
            .ok_or(ProductionMirPlironTranslationErrorV1::KernelShape)?;
        let kir_synchronization = kir_synchronization_contracts_v1(body)?;
        let ranked_synchronization = ranked_synchronization_contracts_v1(recipe)?;
        if kir_synchronization != ranked_synchronization {
            return Err(ProductionMirPlironTranslationErrorV1::SynchronizationMismatch);
        }
        let tensor_operations = if let Some(count) = helpers.nominal_tensors(function, recipe)? {
            count
        } else {
            let kir_tensors = kir_tensor_contracts_v1(body)?;
            let ranked_tensors = ranked_tensor_contracts_v1(recipe)?;
            if kir_tensors != ranked_tensors {
                return Err(ProductionMirPlironTranslationErrorV1::TensorContractMismatch);
            }
            kir_tensors.len()
        };
        Ok((kir_synchronization.len(), tensor_operations))
    }
    fn check_effect_control_flow(
        &mut self,
        body: &FunctionBody,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        locations: &[TranslationEffectLocationV18],
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<(), ProductionMirPlironTranslationErrorV1> {
        let mut total_fallback_reads = BTreeSet::new();
        if let Some(semantic) = self.semantic {
            let function = semantic
                .functions()
                .get(self.function.index() as usize)
                .ok_or(ProductionMirPlironTranslationErrorV1::KernelShape)?;
            for (site, _, _, _) in locations {
                budget
                    .charge()
                    .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
                if site.statement.is_some() || site.ordinal != 0 {
                    continue;
                }
                let Some(block) = function.blocks().get(site.block as usize) else {
                    return Err(ProductionMirPlironTranslationErrorV1::KernelShape);
                };
                let SemanticTerminatorKindV1::Call(call) = block.terminator().kind() else {
                    continue;
                };
                if matches!(
                    semantic.callables().get(call.callee().index() as usize),
                    Some(SemanticCallableDeclV1::CompilerIntrinsic {
                        operation: SemanticCompilerIntrinsicOperationV1::StridedReadView2DLoadOr { .. },
                        ..
                    })
                ) {
                    total_fallback_reads.insert(*site);
                }
            }
        }
        validate_effect_control_flow_v1(body, recipe, locations, &total_fallback_reads, budget)
    }
    fn semantic_sha256(&self) -> &[u8; 32] {
        self.correspondence.semantic_sha256()
    }
    fn diagnostic_subject(&self) -> (SemanticFunctionIdV1, SemanticFunctionIdV1) {
        (self.owner, self.function)
    }
    fn effect(
        &mut self,
        consumer: KirMemoryConsumerV1,
        kir: &dyn KirCorrelationGraphV18,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<TranslationEffectDispositionV18, ProductionMirPlironTranslationErrorV1> {
        if consumer.memory_space == dialect_kernel::MemorySpaceAttr::Private {
            let retained = compiler_owned_retained_local_storage_access_v1(
                consumer,
                kir,
                self.correspondence,
                self.owner,
                self.function,
                &mut BTreeSet::new(),
                budget,
            )
            .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
            let enum_payload = !retained
                && compiler_owned_enum_payload_access_v1(
                    consumer.pointer,
                    kir,
                    self.correspondence,
                    &mut BTreeSet::new(),
                    budget,
                )
                .ok_or(ProductionMirPlironTranslationErrorV1::ResourceLimit)?;
            if retained || enum_payload {
                return Ok(TranslationEffectDispositionV18::LegacyCompilerStorage);
            }
        }
        let site = self
            .sites
            .get(&(consumer.location, consumer.operation_access_ordinal))
            .copied()
            .ok_or(
                ProductionMirPlironTranslationErrorV1::UnattributedExecutableEffect {
                    location: consumer.location,
                },
            )?;
        Ok(TranslationEffectDispositionV18::Ranked { site, consumer })
    }

    fn source_parameter(
        &mut self,
        function: &Function,
        parameter: u32,
        location: FunctionOperationLocation,
    ) -> Result<u32, ProductionMirPlironTranslationErrorV1> {
        let mismatch =
            || ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch { location };
        let Some(semantic) = self.semantic else {
            return Ok(parameter);
        };
        let semantic_function = semantic
            .functions()
            .get(self.function.index() as usize)
            .ok_or_else(mismatch)?;
        let value = function
            .body
            .as_ref()
            .and_then(|body| body.parameters.get(parameter as usize))
            .copied()
            .ok_or_else(mismatch)?;
        semantic_source_argument_for_kir_parameter_v1(
            self.correspondence,
            self.owner,
            semantic_function,
            value,
        )
        .ok_or_else(mismatch)
    }

    fn check_private_access(
        &mut self,
        recipe: &fe2o3_pliron::ProductionRankedKernelV1,
        source: &IndexedRankedAccessSourceV1,
        consumer: KirMemoryConsumerV1,
        site: SemanticAccessSiteV1,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<(), ProductionMirPlironTranslationErrorV1> {
        self.private_arrays
            .as_ref()
            .ok_or(
                ProductionMirPlironTranslationErrorV1::AllocationOriginMismatch {
                    location: consumer.location,
                },
            )?
            .check(recipe, source, consumer, site, budget)
    }

    fn requires_ranked_consumption(
        &mut self,
        site: SemanticAccessSiteV1,
        source: &IndexedRankedAccessSourceV1,
        private: bool,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Result<bool, ProductionMirPlironTranslationErrorV1> {
        if !private {
            return Ok(true);
        }
        match &self.private_arrays {
            Some(relation) => relation.requires_consumption(site, source, budget),
            None => Ok(false),
        }
    }

    fn legacy_private_recipe_exclusion(&self) -> bool {
        true
    }
}

struct TranslationValidationV18 {
    translation: ProductionMirPlironTranslationValidationV1,
    pending_source_storage_effects: usize,
}

impl KirCorrelationGraphV18 for KirCorrelationIndexV1<'_> {
    fn unique_origin(
        &self,
        value: ValueId,
        budget: &mut dyn CorrelationChargeV18,
    ) -> Option<ValueId> {
        walk_unique_kir_ssa_origin_v1(self, value, budget)
    }
    fn operation(&self, location: FunctionOperationLocation) -> Option<&Operation> {
        self.operations.get(&location).copied()
    }
    fn definition(&self, value: ValueId) -> Option<&Operation> {
        self.definitions.get(&value).copied()
    }
    fn definition_location(&self, value: ValueId) -> Option<FunctionOperationLocation> {
        self.definition_locations.get(&value).copied()
    }
    fn block_operations(&self, block: BlockId) -> Option<&[Operation]> {
        self.blocks.get(&block).copied()
    }
    fn inline_scalar(&self) -> &Gfx942InlineScalarCorrespondenceV30<'_> {
        &self.inline_scalar
    }
    fn visit_incoming(
        &self,
        value: ValueId,
        visit: &mut dyn FnMut(ValueId) -> Option<()>,
    ) -> Option<Option<usize>> {
        let Some(inputs) = self.block_parameter_inputs.get(&value) else {
            return Some(None);
        };
        for input in inputs {
            visit(*input)?;
        }
        Some(Some(inputs.len()))
    }
}

struct CorrelationLedgerV18<'b, 'w, 'c> {
    budget: std::cell::RefCell<&'b mut ArgumentBudgetV1<'w>>,
    cleanup: &'c ScopedSourceCleanupV29,
    slot: usize,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    floor: usize,
    failure: std::cell::Cell<Option<ArgumentResourceV1>>,
    inconsistent_inventory: std::cell::Cell<bool>,
}

impl<'b, 'w, 'c> CorrelationLedgerV18<'b, 'w, 'c> {
    fn new(budget: &'b mut ArgumentBudgetV1<'w>, cleanup: &'c ScopedSourceCleanupV29) -> Self {
        Self {
            slot: budget as *const ArgumentBudgetV1<'_> as usize,
            ledger: budget.work_ledger_identity_v1(),
            floor: budget.storage(),
            budget: std::cell::RefCell::new(budget),
            cleanup,
            failure: std::cell::Cell::new(None),
            inconsistent_inventory: std::cell::Cell::new(false),
        }
    }

    fn fail(&self, error: ArgumentResourceV1) -> ArgumentResourceV1 {
        if self.failure.get().is_none() {
            self.failure.set(Some(error));
        }
        self.failure.get().unwrap_or(error)
    }

    fn with_budget<T>(
        &self,
        action: impl FnOnce(&mut ArgumentBudgetV1<'w>) -> Result<T, ArgumentResourceV1>,
    ) -> Result<T, ArgumentResourceV1> {
        if let Some(error) = self.failure.get() {
            return Err(error);
        }
        if self.inconsistent_inventory.get() {
            // Keep the structural refusal in its own channel. This local
            // sentinel stops work; it does not latch a fictitious resource
            // error or acquire permission to refund any containing scope.
            return Err(ArgumentResourceV1::Accounting);
        }
        let mut borrow = self
            .budget
            .try_borrow_mut()
            .map_err(|_| self.fail(ArgumentResourceV1::Accounting))?;
        let budget = &mut **borrow;
        if self.cleanup.is_denied()
            || budget as *const ArgumentBudgetV1<'_> as usize != self.slot
            || budget.work_ledger_identity_v1() != self.ledger
            || budget.storage() < self.floor
        {
            self.cleanup.deny_refund();
            return Err(self.fail(ArgumentResourceV1::Accounting));
        }
        action(budget).map_err(|error| self.fail(error))
    }

    fn inventory<T>(
        &self,
        action: impl FnOnce(
            &mut ArgumentBudgetV1<'w>,
        ) -> Result<T, fe2o3_kernel_analysis::CanonicalKirInventoryErrorV1>,
    ) -> Option<T> {
        if self.inconsistent_inventory.get() {
            return None;
        }
        match self.with_budget(|budget| Ok(action(budget))).ok()? {
            Ok(value) => Some(value),
            Err(error) => {
                match error {
                    fe2o3_kernel_analysis::CanonicalKirInventoryErrorV1::Resource(error) => {
                        self.fail(error);
                    }
                    fe2o3_kernel_analysis::CanonicalKirInventoryErrorV1::InconsistentOwner => {
                        self.inconsistent_inventory.set(true);
                    }
                }
                None
            }
        }
    }
}

struct SourceCorrelationChargeV18<'q, 'b, 'w, 'c> {
    ledger: &'q CorrelationLedgerV18<'b, 'w, 'c>,
    finite: UnsupportedIndexCorrelationBudgetV1,
    finite_denied: bool,
}

impl CorrelationChargeV18 for SourceCorrelationChargeV18<'_, '_, '_, '_> {
    fn reserve_private_array_scratch(&mut self, bytes: usize) -> Option<()> {
        if self.finite_denied {
            return None;
        }
        self.ledger
            .with_budget(|budget| budget.reserve_storage(bytes))
            .ok()
    }

    fn release_private_array_scratch(&mut self, bytes: usize) -> Option<()> {
        if self.finite_denied {
            return None;
        }
        self.ledger
            .with_budget(|budget| {
                if budget
                    .storage()
                    .checked_sub(bytes)
                    .is_none_or(|remaining| remaining < self.ledger.floor)
                {
                    return Err(ArgumentResourceV1::Accounting);
                }
                budget.release_storage(bytes)
            })
            .ok()
    }

    fn charge_many(&mut self, amount: usize) -> Option<()> {
        if self.finite_denied
            || self.ledger.failure.get().is_some()
            || self.ledger.inconsistent_inventory.get()
        {
            return None;
        }
        let Some(remaining) = self.finite.remaining.checked_sub(amount) else {
            self.finite_denied = true;
            return None;
        };
        self.ledger
            .with_budget(|budget| budget.charge_work(amount))
            .ok()?;
        self.finite.remaining = remaining;
        Some(())
    }
}

struct SourceTranslationChargeV18<'q, 'b, 'w, 'c> {
    charge: SourceCorrelationChargeV18<'q, 'b, 'w, 'c>,
    remaining_nodes: Option<usize>,
    node_credits: usize,
    error: Option<ProductionSourceOwnedViewErrorV18>,
}

impl<'q, 'b, 'w, 'c> SourceTranslationChargeV18<'q, 'b, 'w, 'c> {
    fn new(
        ledger: &'q CorrelationLedgerV18<'b, 'w, 'c>,
        work: usize,
    ) -> SourceOwnedResultV18<Self> {
        ledger.with_budget(|budget| budget.reserve_storage(Self::headers()?))?;
        Ok(Self {
            charge: SourceCorrelationChargeV18 {
                ledger,
                finite: UnsupportedIndexCorrelationBudgetV1 { remaining: work },
                finite_denied: false,
            },
            remaining_nodes: None,
            node_credits: 0,
            error: None,
        })
    }

    fn headers() -> Result<usize, ArgumentResourceV1> {
        type Node = NormalizedScalarExpressionV1;
        argument_sum_v1(&[
            size_of::<Self>(),
            size_of::<SourceOwnedResultV18<Self>>(),
            size_of::<Vec<Node>>(),
            size_of::<Result<Vec<Node>, ProductionSemanticKirErrorV1>>(),
            size_of::<Result<Box<[Node; 1]>, Vec<Node>>>(),
            size_of::<Box<[Node]>>(),
            size_of::<NormalizedScalarNodeV18>(),
            argument_product_v1(2, size_of::<Option<Node>>())?,
            size_of::<ProductionSourceOwnedViewErrorV18>(),
        ])
    }

    fn refuse(&mut self, error: ProductionSourceOwnedViewErrorV18) {
        if let ProductionSourceOwnedViewErrorV18::Resource(error) = &error {
            self.charge.ledger.fail(*error);
        }
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    fn live(&self) -> bool {
        self.error.is_none()
            && !self.charge.finite_denied
            && self.charge.ledger.failure.get().is_none()
            && !self.charge.ledger.inconsistent_inventory.get()
    }
}

impl CorrelationChargeV18 for SourceTranslationChargeV18<'_, '_, '_, '_> {
    fn reserve_private_array_scratch(&mut self, bytes: usize) -> Option<()> {
        if !self.live() {
            return None;
        }
        self.charge.reserve_private_array_scratch(bytes)
    }

    fn release_private_array_scratch(&mut self, bytes: usize) -> Option<()> {
        if !self.live() {
            return None;
        }
        self.charge.release_private_array_scratch(bytes)
    }

    fn charge_many(&mut self, amount: usize) -> Option<()> {
        if !self.live() {
            return None;
        }
        self.charge.charge_many(amount)
    }

    fn begin_normalized_tree(&mut self) -> Option<()> {
        if !self.live() {
            return None;
        }
        self.remaining_nodes = Some(fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_NODES_V2);
        Some(())
    }

    fn normalized_node(&mut self) -> Option<()> {
        if !self.live() {
            return None;
        }
        let Some(remaining) = self.remaining_nodes.and_then(|nodes| nodes.checked_sub(1)) else {
            self.refuse(ProductionSourceOwnedViewErrorV18::Binding(
                "expanded scalar tree node allowance",
            ));
            return None;
        };
        // Construction already pays its ordinary node step. Pay bounded
        // destruction before this additional node can become live as well.
        self.charge_many(1)?;
        self.remaining_nodes = Some(remaining);
        Some(())
    }

    fn boxed_normalized_node(
        &mut self,
        node: NormalizedScalarExpressionV1,
    ) -> Option<NormalizedScalarNodeV18> {
        if !self.live() {
            return None;
        }
        let mut charged = 0usize;
        let result = self
            .charge
            .ledger
            .with_budget(|budget| {
                let before = budget.storage();
                let result = emission_vec_v1::<NormalizedScalarExpressionV1>(1, budget);
                charged = budget
                    .storage()
                    .checked_sub(before)
                    .ok_or(ArgumentResourceV1::Accounting)?;
                Ok(result)
            })
            .ok()?;
        let Some(total) = self.node_credits.checked_add(charged) else {
            self.refuse(ArgumentResourceV1::Arithmetic.into());
            return None;
        };
        self.node_credits = total;
        let mut rows = match result {
            Ok(rows) => rows,
            Err(error) => {
                self.refuse(source_emission_error_v18(error));
                return None;
            }
        };
        if rows.capacity() != 1 || !rows.is_empty() {
            self.refuse(ProductionSourceOwnedViewErrorV18::Binding(
                "normalized node exact allocation capacity",
            ));
            return None;
        }
        rows.push(node);
        if rows.len() != 1 || rows.capacity() != 1 {
            self.refuse(ProductionSourceOwnedViewErrorV18::Binding(
                "normalized node exact boxed length",
            ));
            return None;
        }
        // With len == capacity, this safe conversion cannot shrink/reallocate.
        let boxed: Result<Box<[NormalizedScalarExpressionV1; 1]>, _> = rows.try_into();
        match boxed {
            Ok(boxed) => Some(NormalizedScalarNodeV18(boxed)),
            Err(_) => {
                self.refuse(ProductionSourceOwnedViewErrorV18::Binding(
                    "normalized node boxed array conversion",
                ));
                None
            }
        }
    }

    fn finish_normalized_pair(&mut self) -> Option<()> {
        if !self.live() {
            return None;
        }
        // The shared caller invokes this only after both complete trees drop.
        self.charge
            .ledger
            .with_budget(|budget| budget.release_storage(self.node_credits))
            .ok()?;
        self.node_credits = 0;
        self.remaining_nodes = None;
        Some(())
    }
}

struct InventoryCorrelationV18<'i, 'g, 'q, 'b, 'w, 'c> {
    origins: Option<&'i value_origin_v1::WholeValueOriginsV18<'g>>,
    inventory: &'i fe2o3_kernel_analysis::CanonicalKirInventoryV18<'g>,
    function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
    ledger: &'q CorrelationLedgerV18<'b, 'w, 'c>,
    inline_scalar: &'i Gfx942InlineScalarCorrespondenceV30<'g>,
    scalar_source: Option<&'i dyn SourceScalarNormalizationV18>,
}

// Lifetime erasure only: the sole production implementation below borrows the
// original leaf table and checked root argument rows. No external implementor
// or caller-provided value-to-symbol callback can supply source authority.
trait SourceScalarNormalizationV18 {
    fn leaf(
        &self,
        function: &Function,
        value: ValueId,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Option<NormalizedScalarExpressionV1>>;
    fn argument(
        &self,
        function: &Function,
        argument: u32,
        scalar: ProductionSemanticScalarTypeV2,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<NormalizedScalarExpressionV1>;

    fn binary_overflow(
        &self,
        _: &Function,
        _: ValueId,
        overflow: ProductionOverflowContractV2,
        _: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<ProductionOverflowContractV2> {
        Ok(overflow)
    }
}

impl InventoryCorrelationV18<'_, '_, '_, '_, '_, '_> {
    fn block(&self, id: BlockId) -> Option<&fe2o3_kernel_analysis::CanonicalKirBlockRefV1<'_>> {
        self.ledger
            .inventory(|budget| self.inventory.block_for_id(self.function, id, budget))?
    }

    fn result_coordinate(
        &self,
        value: ValueId,
    ) -> Option<fe2o3_kernel_ir::CanonicalKirOperationCoordinateV1> {
        let definition = self.ledger.inventory(|budget| {
            self.inventory
                .definition_for_value(self.function, value, budget)
        })??;
        match definition.coordinate {
            fe2o3_kernel_ir::CanonicalKirDefinitionCoordinateV1::Result { operation, .. } => {
                Some(operation)
            }
            _ => None,
        }
    }

    fn coordinate_operation(
        &self,
        coordinate: fe2o3_kernel_ir::CanonicalKirOperationCoordinateV1,
    ) -> Option<&Operation> {
        self.ledger
            .with_budget(|budget| budget.charge_work(1))
            .ok()?;
        if coordinate.block.function != self.function {
            return None;
        }
        let function = self.inventory.functions().get(self.function.0 as usize)?;
        let block_index = function
            .blocks
            .start
            .checked_add(coordinate.block.block as usize)?;
        if block_index >= function.blocks.end {
            return None;
        }
        let block = self.inventory.blocks().get(block_index)?;
        if block.coordinate != coordinate.block {
            return None;
        }
        let index = block
            .operations
            .start
            .checked_add(coordinate.operation as usize)?;
        if index >= block.operations.end {
            return None;
        }
        let operation = self.inventory.operations().get(index)?;
        (operation.coordinate == coordinate).then_some(operation.operation)
    }
}

impl KirCorrelationGraphV18 for InventoryCorrelationV18<'_, '_, '_, '_, '_, '_> {
    fn source_inventory_identity(&self) -> Option<usize> {
        Some(self.inventory as *const fe2o3_kernel_analysis::CanonicalKirInventoryV18<'_> as usize)
    }
    fn source_definition_index(&self, value: ValueId) -> Option<usize> {
        self.ledger.inventory(|budget| {
            self.inventory
                .definition_index_for_value(self.function, value, budget)
        })?
    }
    fn scalar_result(
        &self,
        function: &Function,
        _: &Operation,
        value: ValueId,
    ) -> Option<ProductionSemanticScalarTypeV2> {
        self.scalar_type(function, value)
    }

    fn scalar_parameter(&self, function: &Function, value: ValueId) -> Option<Option<u32>> {
        self.ledger
            .with_budget(|budget| budget.charge_work(1))
            .ok()?;
        let original = self.inventory.functions().get(self.function.0 as usize)?;
        if !std::ptr::eq(original.function, function) {
            self.ledger.inconsistent_inventory.set(true);
            return None;
        }
        let definition = self.ledger.inventory(|budget| {
            self.inventory
                .definition_for_value(self.function, value, budget)
        })??;
        match definition.coordinate {
            fe2o3_kernel_ir::CanonicalKirDefinitionCoordinateV1::FunctionArgument {
                function,
                argument,
            } if function == self.function => Some(Some(argument)),
            fe2o3_kernel_ir::CanonicalKirDefinitionCoordinateV1::FunctionArgument { .. } => {
                self.ledger.inconsistent_inventory.set(true);
                None
            }
            _ => Some(None),
        }
    }

    fn scalar_type(
        &self,
        function: &Function,
        value: ValueId,
    ) -> Option<ProductionSemanticScalarTypeV2> {
        self.ledger
            .with_budget(|budget| budget.charge_work(1))
            .ok()?;
        let original = self.inventory.functions().get(self.function.0 as usize)?;
        if !std::ptr::eq(original.function, function) {
            self.ledger.inconsistent_inventory.set(true);
            return None;
        }
        let definition = self.ledger.inventory(|budget| {
            self.inventory
                .definition_for_value(self.function, value, budget)
        })??;
        kir_semantic_scalar_v1(definition.ty)
    }

    fn unique_origin(&self, value: ValueId, _: &mut dyn CorrelationChargeV18) -> Option<ValueId> {
        let Some(origins) = self
            .origins
            .filter(|origins| origins.belongs_to(self.inventory, self.function))
        else {
            self.ledger.inconsistent_inventory.set(true);
            return None;
        };
        self.ledger
            .inventory(|budget| origins.value_origin(value, budget))?
    }
    fn scalar_leaf(
        &self,
        function: &Function,
        value: ValueId,
    ) -> Option<Option<NormalizedScalarExpressionV1>> {
        let Some(source) = self.scalar_source else {
            return Some(None);
        };
        self.source_scalar_query(|budget| source.leaf(function, value, budget))
    }

    fn scalar_binary_overflow(
        &self,
        function: &Function,
        value: ValueId,
        overflow: ProductionOverflowContractV2,
    ) -> Option<ProductionOverflowContractV2> {
        match self.scalar_source {
            Some(source) => self.source_scalar_query(|budget| {
                source.binary_overflow(function, value, overflow, budget)
            }),
            None => Some(overflow),
        }
    }

    fn scalar_argument(
        &self,
        function: &Function,
        argument: u32,
        scalar: ProductionSemanticScalarTypeV2,
    ) -> Option<NormalizedScalarExpressionV1> {
        match self.scalar_source {
            Some(source) => self
                .source_scalar_query(|budget| source.argument(function, argument, scalar, budget)),
            None => Some(NormalizedScalarExpressionV1::Symbol {
                symbol: PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2.checked_add(argument)?,
                scalar,
            }),
        }
    }

    fn operation(&self, location: FunctionOperationLocation) -> Option<&Operation> {
        let block = self.block(location.block)?;
        self.coordinate_operation(fe2o3_kernel_ir::CanonicalKirOperationCoordinateV1 {
            block: block.coordinate,
            operation: u32::try_from(location.operation_index).ok()?,
        })
    }

    fn definition(&self, value: ValueId) -> Option<&Operation> {
        self.coordinate_operation(self.result_coordinate(value)?)
    }

    fn definition_location(&self, value: ValueId) -> Option<FunctionOperationLocation> {
        let coordinate = self.result_coordinate(value)?;
        let function = self.inventory.functions().get(self.function.0 as usize)?;
        let index = function
            .blocks
            .start
            .checked_add(coordinate.block.block as usize)?;
        if index >= function.blocks.end {
            return None;
        }
        self.ledger
            .with_budget(|budget| budget.charge_work(1))
            .ok()?;
        let block = self.inventory.blocks().get(index)?;
        if block.coordinate != coordinate.block {
            return None;
        }
        Some(FunctionOperationLocation::new(
            block.block.id,
            coordinate.operation as usize,
        ))
    }

    fn block_operations(&self, block: BlockId) -> Option<&[Operation]> {
        Some(self.block(block)?.block.operations.as_slice())
    }

    fn inline_scalar(&self) -> &Gfx942InlineScalarCorrespondenceV30<'_> {
        self.inline_scalar
    }

    fn visit_incoming(
        &self,
        value: ValueId,
        visit: &mut dyn FnMut(ValueId) -> Option<()>,
    ) -> Option<Option<usize>> {
        let definition = self.ledger.inventory(|budget| {
            self.inventory
                .definition_index_for_value(self.function, value, budget)
        })??;
        let row = self.inventory.definitions().get(definition)?;
        if !matches!(
            row.coordinate,
            fe2o3_kernel_ir::CanonicalKirDefinitionCoordinateV1::BlockArgument { .. }
        ) {
            return Some(None);
        }
        let function = self.inventory.functions().get(self.function.0 as usize)?;
        let mut count = 0_usize;
        // The inventory has no reverse-edge index. Scan its existing bounded
        // function range, preserving duplicate successor occurrences exactly.
        for row in self
            .inventory
            .edge_arguments()
            .get(function.edge_arguments.clone())?
        {
            self.ledger
                .with_budget(|budget| budget.charge_work(1))
                .ok()?;
            if row.target_definition == definition {
                visit(row.value)?;
                count = count.checked_add(1)?;
            }
        }
        Some(Some(count))
    }
}

impl InventoryCorrelationV18<'_, '_, '_, '_, '_, '_> {
    fn source_scalar_query<T>(
        &self,
        query: impl FnOnce(&mut ArgumentBudgetV1<'_>) -> SourceOwnedResultV18<T>,
    ) -> Option<T> {
        match self.ledger.with_budget(|budget| Ok(query(budget))).ok()? {
            Ok(value) => Some(value),
            Err(ProductionSourceOwnedViewErrorV18::Resource(error)) => {
                self.ledger.fail(error);
                None
            }
            Err(_) => {
                self.ledger.inconsistent_inventory.set(true);
                None
            }
        }
    }
}

// Address-space normalization is a dataflow fact, never an allocation/epoch
// certificate. In particular an original Generic argument remains Unknown.
// Only the checked PointerToGeneric operation introduces a concrete-space
// witness into the Generic graph; transport merely forwards that witness.
struct SourcePointerSpacesV18<'i, 'g> {
    inventory: &'i fe2o3_kernel_analysis::CanonicalKirInventoryV18<'g>,
    function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
    definitions: std::ops::Range<usize>,
    states: Vec<origin_worklist_v1::OriginStateV1<AddressSpace>>,
}

fn address_type_space_v18(ty: &Type) -> Option<AddressSpace> {
    match ty {
        Type::Pointer(pointer) => Some(pointer.address_space),
        Type::Slice(slice) => Some(slice.address_space),
        _ => None,
    }
}

impl<'i, 'g> SourcePointerSpacesV18<'i, 'g> {
    fn derive(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        inventory: &'i fe2o3_kernel_analysis::CanonicalKirInventoryV18<'g>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        relation.query(budget)?;
        if !std::ptr::eq(relation.inventory, inventory) {
            return relation.source.missing("pointer-space inventory changed");
        }
        Self::derive_inner(relation, inventory, function, budget)
    }

    fn derive_optimized(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        optimized: &ProductionOptimizedSourceCorrespondenceV18<'_>,
        inventory: &'i fe2o3_kernel_analysis::CanonicalKirInventoryV18<'g>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        optimized_source_endpoints_v18(relation, optimized, budget)?;
        if !std::ptr::eq(optimized.output_inventory(budget)?, inventory) {
            return relation
                .source
                .missing("optimized pointer-space inventory changed");
        }
        Self::derive_inner(relation, inventory, function, budget)
    }

    fn derive_inner(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        inventory: &'i fe2o3_kernel_analysis::CanonicalKirInventoryV18<'g>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Self> {
        use fe2o3_kernel_ir::CanonicalKirDefinitionCoordinateV1 as Definition;
        use origin_worklist_v1::{
            OriginStateV1 as State, OriginWorkErrorV1 as Error, OriginWorkV1,
        };
        let failure = |error| match error {
            Error::Resource(error) => ProductionSourceOwnedViewErrorV18::Resource(error),
            Error::Shape => {
                ProductionSourceOwnedViewErrorV18::Binding("source pointer-space dataflow")
            }
        };
        let owner = inventory
            .functions()
            .get(function.0 as usize)
            .filter(|row| row.coordinate == function && row.function.body.is_some())
            .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                "pointer-space function",
            ))?;
        let definitions = owner.definitions.clone();
        let rows = inventory.definitions().get(definitions.clone()).ok_or(
            ProductionSourceOwnedViewErrorV18::Binding("pointer-space definitions"),
        )?;
        let mut links = 0usize;
        Self::visit_links(relation, inventory, function, budget, |_, _, budget| {
            budget.charge_work(1)?;
            links = links.checked_add(1).ok_or(ArgumentResourceV1::Arithmetic)?;
            Ok(())
        })?;
        let mut solver = OriginWorkV1::new(rows.len(), links, budget).map_err(failure)?;
        for row in rows {
            budget.charge_work(1)?;
            let seed = match address_type_space_v18(row.ty) {
                None => State::Unknown,
                Some(AddressSpace::Generic) => match row.coordinate {
                    Definition::FunctionArgument { .. } => State::Unknown,
                    Definition::BlockArgument { block, .. } if block.block == 0 => State::Unknown,
                    _ => State::Pending,
                },
                Some(concrete) => State::Exact(concrete),
            };
            solver.seed_next(seed, budget).map_err(failure)?;
        }
        Self::visit_links(
            relation,
            inventory,
            function,
            budget,
            |source, target, budget| solver.add_link(source, target, budget).map_err(failure),
        )?;
        let states = solver.solve(budget).map_err(failure)?;
        Ok(Self {
            inventory,
            function,
            definitions,
            states,
        })
    }

    fn visit_links(
        relation: &ProductionSourceCorrespondenceV18<'_>,
        inventory: &fe2o3_kernel_analysis::CanonicalKirInventoryV18<'_>,
        function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
        budget: &mut ArgumentBudgetV1<'_>,
        mut visit: impl FnMut(usize, usize, &mut ArgumentBudgetV1<'_>) -> SourceOwnedResultV18<()>,
    ) -> SourceOwnedResultV18<()> {
        use fe2o3_kernel_ir::CanonicalKirDefinitionCoordinateV1 as Definition;
        let owner = inventory
            .functions()
            .get(function.0 as usize)
            .filter(|row| row.coordinate == function && row.function.body.is_some())
            .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                "pointer-space function",
            ))?;
        let definitions = owner.definitions.clone();
        let rows = inventory.definitions().get(definitions.clone()).ok_or(
            ProductionSourceOwnedViewErrorV18::Binding("pointer-space definitions"),
        )?;
        let edges = inventory
            .edge_arguments()
            .get(owner.edge_arguments.clone())
            .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                "pointer-space edge arguments",
            ))?;
        for (target, row) in rows.iter().enumerate() {
            budget.charge_work(1)?;
            if address_type_space_v18(row.ty) != Some(AddressSpace::Generic) {
                continue;
            }
            let Definition::Result { operation, result } = row.coordinate else {
                continue;
            };
            let block = owner
                .blocks
                .start
                .checked_add(operation.block.block as usize)
                .filter(|index| *index < owner.blocks.end)
                .and_then(|index| inventory.blocks().get(index))
                .filter(|row| row.coordinate == operation.block)
                .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                    "pointer-space result block",
                ))?;
            let actual = block
                .operations
                .start
                .checked_add(operation.operation as usize)
                .filter(|index| *index < block.operations.end)
                .and_then(|index| inventory.operations().get(index))
                .filter(|row| row.coordinate == operation)
                .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                    "pointer-space result operation",
                ))?;
            if actual
                .operation
                .results
                .get(result as usize)
                .is_none_or(|value| Some(value.id) != row.value || &value.ty != row.ty)
            {
                return relation.source.missing("pointer-space result identity");
            }
            let inputs = match &actual.operation.kind {
                OperationKind::Cast {
                    kind: CastKind::SliceToGeneric,
                    value,
                    to,
                } => {
                    let input = inventory
                        .definition_for_value(function, *value, budget)
                        .map_err(source_pointer_inventory_error_v18)?
                        .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                            "Generic descriptor input",
                        ))?;
                    if to != row.ty || !source_descriptor_widening_v29(input.ty, to) {
                        return relation
                            .source
                            .missing("Generic descriptor changes its checked contract");
                    }
                    [Some(*value), None]
                }
                OperationKind::Cast {
                    kind: CastKind::PointerToGeneric,
                    value,
                    to,
                } => {
                    let input = inventory
                        .definition_for_value(function, *value, budget)
                        .map_err(source_pointer_inventory_error_v18)?
                        .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                            "Generic cast input",
                        ))?;
                    let (Type::Pointer(input), Type::Pointer(output)) = (input.ty, to) else {
                        return relation
                            .source
                            .missing("Generic cast is not a pointer exposure");
                    };
                    if to != row.ty
                        || output.address_space != AddressSpace::Generic
                        || input.address_space == AddressSpace::Generic
                        || input.pointee != output.pointee
                        || input.access != output.access
                        || (input.address_space == AddressSpace::Constant
                            && input.access != AccessMode::ReadOnly)
                    {
                        return relation
                            .source
                            .missing("Generic cast changes its checked contract");
                    }
                    [Some(*value), None]
                }
                OperationKind::Cast {
                    kind: CastKind::RestrictPointerAccess,
                    value,
                    ..
                }
                | OperationKind::GetElementPointer { base: value, .. }
                | OperationKind::SliceData { slice: value }
                | OperationKind::Storage(fe2o3_kernel_ir::StorageOperationV1::Project {
                    base: value,
                    ..
                }) => {
                    source_generic_transport_input_v18(inventory, function, *value, budget)?;
                    [Some(*value), None]
                }
                OperationKind::Select {
                    true_value,
                    false_value,
                    ..
                } => {
                    source_generic_transport_input_v18(inventory, function, *true_value, budget)?;
                    source_generic_transport_input_v18(inventory, function, *false_value, budget)?;
                    [Some(*true_value), Some(*false_value)]
                }
                // A load, integer/bit cast or call result cannot introduce a
                // concrete-space fact without its separate source receipt.
                _ => [None, None],
            };
            for value in inputs.into_iter().flatten() {
                let source = inventory
                    .definition_index_for_value(function, value, budget)
                    .map_err(source_pointer_inventory_error_v18)?
                    .filter(|index| definitions.contains(index))
                    .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                        "pointer-space source definition",
                    ))?;
                visit(source - definitions.start, target, budget)?;
            }
        }
        for edge in edges {
            budget.charge_work(1)?;
            if !definitions.contains(&edge.incoming_definition)
                || !definitions.contains(&edge.target_definition)
            {
                return relation
                    .source
                    .missing("pointer-space edge changed function");
            }
            let target = edge.target_definition - definitions.start;
            if address_type_space_v18(rows[target].ty) != Some(AddressSpace::Generic) {
                continue;
            }
            if !matches!(rows[target].coordinate, Definition::BlockArgument { .. })
                || rows[edge.incoming_definition - definitions.start].ty != rows[target].ty
            {
                return relation
                    .source
                    .missing("pointer-space edge type or destination");
            }
            visit(edge.incoming_definition - definitions.start, target, budget)?;
        }
        Ok(())
    }

    fn concrete_space(
        &self,
        value: ValueId,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> SourceOwnedResultV18<Option<AddressSpace>> {
        let index = self
            .inventory
            .definition_index_for_value(self.function, value, budget)
            .map_err(source_pointer_inventory_error_v18)?
            .filter(|index| self.definitions.contains(index))
            .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
                "pointer-space query definition",
            ))?;
        budget.charge_work(1)?;
        match self.states.get(index - self.definitions.start) {
            Some(origin_worklist_v1::OriginStateV1::Exact(space))
                if *space != AddressSpace::Generic =>
            {
                Ok(Some(*space))
            }
            Some(origin_worklist_v1::OriginStateV1::Unknown) => Ok(None),
            _ => Err(ProductionSourceOwnedViewErrorV18::Binding(
                "unsolved pointer-space query",
            )),
        }
    }
}

fn source_pointer_inventory_error_v18(
    error: fe2o3_kernel_analysis::CanonicalKirInventoryErrorV1,
) -> ProductionSourceOwnedViewErrorV18 {
    match error {
        fe2o3_kernel_analysis::CanonicalKirInventoryErrorV1::Resource(error) => error.into(),
        _ => ProductionSourceOwnedViewErrorV18::Binding("pointer-space inventory query"),
    }
}

fn source_generic_transport_input_v18(
    inventory: &fe2o3_kernel_analysis::CanonicalKirInventoryV18<'_>,
    function: fe2o3_kernel_ir::CanonicalKirFunctionCoordinateV1,
    value: ValueId,
    budget: &mut ArgumentBudgetV1<'_>,
) -> SourceOwnedResultV18<()> {
    let definition = inventory
        .definition_for_value(function, value, budget)
        .map_err(source_pointer_inventory_error_v18)?
        .ok_or(ProductionSourceOwnedViewErrorV18::Binding(
            "Generic transport input",
        ))?;
    if address_type_space_v18(definition.ty) != Some(AddressSpace::Generic) {
        return Err(ProductionSourceOwnedViewErrorV18::Binding(
            "Generic transport is not a checked exposure",
        ));
    }
    Ok(())
}
