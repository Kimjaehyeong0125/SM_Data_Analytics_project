from datapkg.io import load_population, save_population
from utilpkg.stats import zscore_columns
from scoringpkg.aggregate import compute_total_score
from scoringpkg.pca_model import fit_pca_on_unstructured
from scoringpkg.cutoff import compute_cutoff_from_accepted
from servicepkg.user_service import evaluate_new_applicants


def main():
    pop_df = load_population("population_scores.csv")

    structured_cols = [
        "교과등급평균",
        "전공교과평균",
        "국어평균",
        "수학평균",
        "영어평균",
        "과학평균",
        "수상개수",
        "활동개수",
        "봉사시간",
    ]

    unstructured_cols = [
        "학업성취도",
        "학업태도",
        "탐구력",
        "전공교과성취",
        "전공세특",
        "진로탐색",
        "협력소통",
        "나눔배려",
        "성실규칙",
        "리더십",
    ]

    pop_df = zscore_columns(pop_df, structured_cols + unstructured_cols)
    structured_z_cols = [c + "_z" for c in structured_cols]
    unstructured_z_cols = [c + "_z" for c in unstructured_cols]

    pop_df, pca_model = fit_pca_on_unstructured(pop_df, unstructured_z_cols, n_components=2)

    pop_df = compute_total_score(
        pop_df,
        structured_z_cols=structured_z_cols,
        unstructured_z_cols=unstructured_z_cols,
        w_structured=0.6,
        w_unstructured=0.4,
    )

    cutoff_info = compute_cutoff_from_accepted(pop_df, trim_ratio=0.1)
    print("Cutoff info:", cutoff_info)

    cutoff_score = cutoff_info["trimmed_cutoff"]

    try:
        result_new = evaluate_new_applicants(
            pop_df,
            "new_applicant.csv",
            structured_cols=structured_cols,
            unstructured_cols=unstructured_cols,
            cutoff_score=cutoff_score,
        )

        print("\n[새로운 지원자 평가 결과]")
        print(result_new[["s_id", "total_score", "pass_by_cutoff"]])
    except FileNotFoundError:
        print("\nnew_applicant.csv 파일이 없어서 신규 지원자 평가는 건너뜁니다.")

    save_population("population_scores_with_result.csv", pop_df)
    print("\npopulation_scores_with_result.csv 파일로 저장 완료.")


if __name__ == "__main__":
    main()
