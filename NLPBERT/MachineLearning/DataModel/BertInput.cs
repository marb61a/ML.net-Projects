namespace NLPBERT.MachineLearning.DataModel
{
    public class BertInput
    {
        [VectorType(1, 256)]
        [ColumnName("input_ids:0")]
        public long[] InputId { get; set; }

        [VectorType(1, 256)]
        [ColumnName("input_mask:0")]
        public long[] InputMask { get; set; }

        [VectorType(1, 256)]
        [ColumnName("segment_ids:0")]
        public long[] SegmentIds { get; set; }

        [VectorType(1)]
        [ColumnName("unique_ids_raw_output___9:0")]
        public long[] UniqueIds { get; set; }

    }
}